#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import calc_ipv as cpv
import thermal_tropopause as tpp
from scipy import interpolate
from scipy import signal as sig

from numpy.polynomial import chebyshev as cby
from numpy.polynomial import legendre
from numpy.polynomial import laguerre
from numpy.polynomial import polynomial

import cmip5.common.staticParams as sp
plt.style.use('ggplot')


def log_setup(name, out_file):
    """
    Create a logger object with name and file location.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    lfh = logging.FileHandler(out_file)
    lfh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    lfh.setFormatter(formatter)

    logger.addHandler(lfh)
    return logger


def get_data(year, tidx_s=0, tidx_e=None, hemis='SH', root_dir='/Volumes/FN_2187/erai'):

    if 'monthly' in root_dir:
        time_skip = None
        lat_skip = 3
    else:
        time_skip = None
        lat_skip = None

    in_file = '{}/erai_theta_{:04d}.nc'.format(root_dir, year)
    data = nc.Dataset(in_file, 'r')
    lat = data.variables['latitude'][:]
    lat_0 = 5.0
    if hemis == 'SH':
        lat_sel = lat < -lat_0
    else:
        lat_sel = lat > lat_0

    pv = data.variables['pv'][tidx_s:tidx_e:time_skip, :, lat_sel, ...]
    uwnd = data.variables['u'][tidx_s:tidx_e:time_skip, :, lat_sel, ...]
    vwnd = data.variables['v'][tidx_s:tidx_e:time_skip, :, lat_sel, ...]
    pres = data.variables['pres'][tidx_s:tidx_e:time_skip, :, lat_sel, ...]
    lat = lat[lat_sel]

    time = data.variables['time'][tidx_s:tidx_e:time_skip]

    if lat_skip is not None:
        pv = pv[:, :, ::lat_skip, :]
        uwnd = uwnd[:, :, ::lat_skip, :]
        pres = pres[:, :, ::lat_skip, :]
        lat = lat[::lat_skip]

    time_units = data.variables['time'].units
    lon = data.variables['longitude'][:]
    theta = data.variables['level'][:]

    return {'pv': pv * 1e6, 'pres': pres, 'uwnd': uwnd, 'vwnd': vwnd,
            'lat': lat, 'lon': lon, 'lev': theta, 'time': time, 'tunits': time_units}


def get_data_merra(year, tidx_s, tidx_e):
    in_file = '/Volumes/FN_2187/merra/merra_{:04d}.nc'.format(year)
    data = nc.Dataset(in_file, 'r')
    lat = data.variables['YDim'][:]
    lat_0 = 10.0
    pv = data.variables['EPV'][tidx_s:tidx_e, :, lat < lat_0, ...]
    uwnd = data.variables['U'][tidx_s:tidx_e, :, lat < lat_0, ...]
    t_air = data.variables['T'][tidx_s:tidx_e, :, lat < lat_0, ...]

    pres = data.variables['Height'][:]
    theta = cpv.theta(t_air, pres)

    pv_theta = cpv.vinterp(pv, theta, sp.th_levels)
    u_theta = cpv.vinterp(uwnd, theta, sp.th_levels)
    pres_theta = cpv.vinterp(pres, theta, sp.th_levels)

    time = data.variables['TIME'][tidx_s:tidx_e]
    time_units = data.variables['TIME'].units

    lat = lat[lat < lat_0]
    lon = data.variables['XDim'][:]
    # theta = data.variables['level'][:]

    return {'pv': pv_theta * 1e6, 'pres': pres_theta, 'uwnd': u_theta,
            'lat': lat, 'lon': lon, 'lev': sp.th_levels, 'time': time,
            'tunits': time_units}


def main():
    p_0 = 100000.0
    kppa = 287.0 / 1004.0
    year = 2000
    therm_trop = False
    high_res = False
    monthly_plots = False
    fit_type = 'poly'

    fd_jet = False
    hemis = 'SH'

    max_lev = 400  # np.max(theta_xpv)
    fit_deg = 6

    if hemis == 'SH':
        hmult = 1
    else:
        hmult = -1

    year_s = 1979
    year_e = 2016

    jet_loc_ts = []
    mon_idx = 0
    debug_log = log_setup('jet_find', './find_jet_cby.log')

    if fit_type == 'cheb':
        pfit = cby.chebfit
        pder = cby.chebder
        peval = cby.chebval
    elif fit_type == 'legendre':
        pfit = legendre.legfit
        pder = legendre.legder
        peval = legendre.legval
    elif fit_type == 'laguerre':
        pfit = laguerre.lagfit
        pder = laguerre.lagder
        peval = laguerre.lagval
    elif fit_type == 'poly':
        pfit = polynomial.polyfit
        pder = polynomial.polyder
        peval = polynomial.polyval


    for year in range(year_s, year_e + 1):
        debug_log.info('CALCULATE: {}'.format(year))
        data = get_data(year, root_dir='/Volumes/FN_2187/erai/monthly', hemis=hemis)
        lat, lon, lev = data['lat'], data['lon'], data['lev']
        dates = nc.num2date(data['time'], data['tunits'])

        pv_mean = np.mean(data['pv'], axis=-1)

        if therm_trop:
            debug_log.info('FIND THERMAL TROP')
            #t_air = lev[None, :, None, None] / (p_0 / data['pres'])**kppa
            #pres_levs = np.logspace(5, 3, lev.shape[0])
            #print(pres_levs)
            #t_pres = cpv.vinterp(t_air, data['pres'], pres_levs)
            #trop_temp, trop_pres = tpp.get_tropopause_pres(t_pres, pres_levs)
            trop_temp, trop_pres = tpp.get_tropopause_theta(lev, data['pres'])
            trop_theta = np.nanmean(cpv.theta(trop_temp, trop_pres), axis=-1)
        theta_xpv = cpv.vinterp(data['lev'], data['pv'], np.array([-2.0 * hmult]))
        theta_xpv = np.nanmean(theta_xpv, axis=-1)

        #pv_levs = np.linspace(-1.5, -2.5, 10)
        #theta_npv = cpv.vinterp(data['lev'], data['pv'], pv_levs)
        #theta_npv = np.nanmean(theta_npv, axis=-1)

        if high_res:
            lat_hr = np.linspace(lat.min(), lat.max(), lat.shape[0] * 2)
            theta_hr = interpolate.interp1d(lat, theta_xpv, axis=-1)(lat_hr)


        for idx in range(pv_mean.shape[0]):
            y_s = 0
            y_e = None
            if therm_trop:
                y_s = np.abs(trop_theta[idx, np.abs(lat) < 50] -
                             theta_xpv[idx, np.abs(lat) < 50]).argmin()
                y_e = np.abs(80.0 - np.abs(lat)).argmin()

                if lat[0] < lat[-1]:
                    y_s, y_e = y_e, y_s
                if y_e == 0:
                    y_e = -1

                if y_s is None:
                    debug_log.info('INTERSECTION AT: {} (:{}, {})'
                                   .format(lat[y_e], y_e, lat.shape))
                else:
                    debug_log.info('INTERSECTION AT: {} ({}:, {})'
                                   .format(lat[y_s], y_s, lat.shape))

            theta_cby_fit = pfit(lat[y_s:y_e], theta_xpv[idx, y_s:y_e], fit_deg)
            dtdphi_cby = pder(theta_cby_fit)

            theta_cby = peval(lat[y_s:y_e], theta_cby_fit)
            dtheta_cby = peval(lat[y_s:y_e], dtdphi_cby)

            d2theta = pder(theta_cby_fit, 2)
            d2theta = peval(lat[y_s:y_e], d2theta)

            theta_r = theta_xpv[idx, y_s:y_e]
            lat_r = lat[y_s:y_e]
            dtheta_fd = (theta_r[2:] - theta_r[:-2]) / (lat_r[2:] - lat_r[:-2])

            #if idx == 9:
            #    import pdb;pdb.set_trace()

            # rel_min = set(sig.argrelmin(dtheta_cby)[0])
            # d2_zeros = set(sig.argrelmin(np.abs(d2theta))[0])
            # jet_loc = list(rel_min.intersection(d2_zeros))


            if fd_jet:
                jet_loc = sig.argrelmax(hmult * dtheta_fd)[0].astype(int)
            else:
                jet_loc = sig.argrelmax(hmult * dtheta_cby)[0].astype(int)

            if y_s is None:
                jet_loc -= y_e
            else:
                jet_loc += y_s

            if len(jet_loc) == 0:
                debug_log.info("{0} NO LOC {1}-{2:02d} {0}".format('-' * 20, year,
                                                                   idx + 1))
                jet_loc_ts.append(0)

            elif len(jet_loc) == 1:
                jet_loc_ts.append(jet_loc[0])

            elif len(jet_loc) > 1:
                jet_loc_ts.append(jet_loc[np.abs(lat[jet_loc]).argmin()])

            if abs(lat[int(jet_loc_ts[mon_idx])]) > 45.0:
                debug_log.info('JET POLEWARD OF 45: {} {:02d}'.format(year, idx + 1))
                monthly_plots_temp = True
            else:
                monthly_plots_temp = monthly_plots

            if monthly_plots_temp:
                fig, axis = plt.subplots(1, 1, figsize=(15, 15))
                ax2 = axis.twinx()
                #axis.contourf(lat, lev[lev <= max_lev],
                #              np.mean(data['uwnd'][idx, lev <= max_lev, ...], axis=-1),
                #              np.linspace(-40, 40, 21), cmap='RdBu_r', extend='both')

                axis.pcolormesh(lat, lev[lev <= max_lev],
                                np.mean(data['uwnd'][idx, lev <= max_lev, ...], axis=-1),
                                vmin=-40, vmax=40, cmap='RdBu_r')


                #for lev_ix, pv_lev in enumerate(pv_levs):
                #    axis.plot(lat, theta_npv[idx, lev_ix, :], label=pv_lev)fit_deg

                axis.plot(lat[:None], theta_xpv[idx, :None], 'kx-')
                if therm_trop:
                    axis.plot(lat[:None], trop_theta[idx, :None], 'g-')
                    if y_e is None:
                        axis.plot(lat[y_s], theta_xpv[idx, y_s], 'ko')
                        axis.plot(lat[y_s], trop_theta[idx, y_s], 'go')
                    else:
                        axis.plot(lat[y_e], theta_xpv[idx, y_e], 'ko')
                        axis.plot(lat[y_e], trop_theta[idx, y_e], 'go')

                axis.plot(lat[y_s:y_e], theta_cby)
                ax2.plot(lat[y_s:y_e], dtheta_cby, 'g')
                ax2.plot(lat_r[1:-1], dtheta_fd, 'C1x-')

                if y_s is not None:
                    corr = -y_s
                else:
                    corr = 0
                if fd_jet:
                    ax2.plot(lat[jet_loc], dtheta_fd[jet_loc + corr], 'C1o')
                else:
                    ax2.plot(lat[jet_loc], dtheta_cby[jet_loc + corr], 'go')

                #axis.plot(2 * [lat[dtheta_cby[1:-1].argmin() + 1]], axis.get_ylim(), '--')

                axis.plot(2 * [lat[int(jet_loc_ts[mon_idx])]], axis.get_ylim(), '--')
                axis.plot(lat[int(jet_loc_ts[mon_idx])],
                          theta_xpv[idx, int(jet_loc_ts[mon_idx])], 'kx', ms=7)

                #ax2.set_ylim([-4, 4])
                ax2.grid(b=False)
                axis.set_ylim([270, 400])
                #plt.colorbar()
                axis.set_title('{:%Y-%m-%d %H}'.format(dates[idx]))
                if fd_jet:
                    plt.savefig('plt_pv_fd_{}_{:04d}.png'.format(year, idx))
                else:
                    plt.savefig('plt_pv_cby_{}_{:04d}.png'.format(year, idx))
                plt.tight_layout()
                #plt.show()
                plt.close()

            mon_idx += 1
    #axis.legend()
    fig, ax = plt.subplots(1, 1, figsize=(19, 5))
    ax.plot(lat[np.array(jet_loc_ts).astype(int)], 'x-')
    plt.tight_layout()
    if fd_jet:
        plt.savefig('plt_jet_loc_ts_{}_fd_{}-{}.pdf'.format(hemis, year_s, year_e))
    else:
        plt.savefig('plt_jet_loc_ts_{}_{}{}_{}-{}.pdf'.format(hemis, fit_type, fit_deg,
                                                              year_s, year_e))

    plt.show()


if __name__ == "__main__":
    main()
