import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import calc_ipv as cpv
import thermal_tropopause as tpp
from scipy import interpolate
from scipy import signal as sig

from numpy.polynomial import chebyshev as cby
from numpy.polynomial import legendre

import cmip5.common.staticParams as sp
import cmip5.common.atmos as atm


plt.style.use('ggplot')


def get_data(year, tidx_s=0, tidx_e=None, root_dir='/Volumes/FN_2187/erai'):

    if 'monthly' in root_dir:
        time_skip = None
        lat_skip = 3
    else:
        time_skip = 2
        lat_skip = None

    in_file = '{}/erai_theta_{:04d}.nc'.format(root_dir, year)
    data = nc.Dataset(in_file, 'r')
    lat = data.variables['latitude'][:]
    lat_0 = 10.0
    pv = data.variables['pv'][tidx_s:tidx_e:time_skip, :, lat > lat_0, ...]
    uwnd = data.variables['u'][tidx_s:tidx_e:time_skip, :, lat > lat_0, ...]
    pres = data.variables['pres'][tidx_s:tidx_e:time_skip, :, lat > lat_0, ...]
    time = data.variables['time'][tidx_s:tidx_e:time_skip]

    if lat_skip is not None:
        pv = pv[:, :, ::lat_skip, :]
        uwnd = uwnd[:, :, ::lat_skip, :]
        pres = pres[:, :, ::lat_skip, :]
        lat = lat[::lat_skip]

    time_units = data.variables['time'].units
    lat = lat[lat > lat_0]
    lon = data.variables['longitude'][:]
    theta = data.variables['level'][:]

    return {'pv': pv * 1e6, 'pres': pres, 'uwnd': uwnd,
            'lat': lat, 'lon': lon, 'lev': theta, 'time': time, 'tunits': time_units}


def get_data_merra(year, tidx_s, tidx_e):
    in_file = '/Volumes/FN_2187/merra/merra_{:04d}.nc'.format(year)
    data = nc.Dataset(in_file, 'r')
    lat = data.variables['YDim'][:]
    lat_0 = 10.0
    pv = data.variables['EPV'][tidx_s:tidx_e, :, lat > lat_0, ...]
    uwnd = data.variables['U'][tidx_s:tidx_e, :, lat > lat_0, ...]
    t_air = data.variables['T'][tidx_s:tidx_e, :, lat > lat_0, ...]

    pres = data.variables['Height'][:]
    theta = cpv.theta(t_air, pres)

    pv_theta = cpv.vinterp(pv, theta, sp.th_levels)
    u_theta = cpv.vinterp(uwnd, theta, sp.th_levels)
    pres_theta = cpv.vinterp(pres, theta, sp.th_levels)

    time = data.variables['TIME'][tidx_s:tidx_e]
    time_units = data.variables['TIME'].units

    lat = lat[lat > lat_0]
    lon = data.variables['XDim'][:]
    # theta = data.variables['level'][:]

    return {'pv': pv_theta * 1e6, 'pres': pres_theta, 'uwnd': u_theta,
            'lat': lat, 'lon': lon, 'lev': sp.th_levels, 'time': time,
            'tunits': time_units}


if __name__ == "__main__":
    p_0 = 100000.0
    kppa = 287.0 / 1004.0
    year = 2000
    therm_trop = False
    high_res = False
    monthly_plots = False

    year_s = 1979
    year_e = 2016

    jet_loc_ts = np.zeros((year_e - year_s + 1) * 12)
    mon_idx = 0

    #pfit = cby.chebfit
    #pder = cby.chebder
    #peval = cby.chebval

    pfit = legendre.legfit
    pder = legendre.legder
    peval = legendre.legval

    for year in range(year_s, year_e + 1):

        data = get_data(year, root_dir='/Volumes/FN_2187/erai')
        lat, lon, lev = data['lat'], data['lon'], data['lev']
        dates = nc.num2date(data['time'], data['tunits'])

        pv_mean = np.mean(data['pv'], axis=-1)

        if therm_trop:
            t_air = lev[None, :, None, None] / (p_0 / data['pres'])**kppa
            pres_levs = np.logspace(5, 3, 16)
            t_pres = cpv.vinterp(t_air, data['pres'], pres_levs)
            trop_temp, trop_pres = tpp.get_tropopause(t_pres, pres_levs)
            trop_theta = cpv.theta(trop_temp, trop_pres)

        theta_xpv = cpv.vinterp(data['lev'], data['pv'], np.array([2.0]))
        theta_xpv = np.nanmean(theta_xpv, axis=-1)

        if high_res:
            lat_hr = np.linspace(lat.min(), lat.max(), lat.shape[0] * 2)
            theta_hr = interpolate.interp1d(lat, theta_xpv, axis=-1)(lat_hr)

        max_lev = 400  # np.max(theta_xpv)
        fit_deg = 10

        #for idx in range(5, pv_mean.shape[0]):
        for idx in range(0, 12):
        #idx = 0

            theta_cby_fit = pfit(lat[:None], theta_xpv[idx, :None], fit_deg)
            dtdphi_cby = pder(theta_cby_fit)

            theta_cby = peval(lat[:None], theta_cby_fit)
            dtheta_cby = peval(lat[:None], dtdphi_cby)

            d2theta = pder(theta_cby_fit, 2)
            d2theta = peval(lat, d2theta)

            # rel_min = set(sig.argrelmin(dtheta_cby)[0])
            # d2_zeros = set(sig.argrelmin(np.abs(d2theta))[0])
            # jet_loc = list(rel_min.intersection(d2_zeros))

            jet_loc = sig.argrelmin(dtheta_cby)[0].astype(int)

            if len(jet_loc) == 0:
                print("{0} NO LOC {1}-{2} {0}".format('-' * 20, year, idx + 1))

            elif len(jet_loc) == 1:
                jet_loc_ts[mon_idx] = jet_loc[0]

            elif len(jet_loc) > 1:
                jet_loc_ts[mon_idx] = jet_loc[lat[jet_loc].argmin()]

            if lat[int(jet_loc_ts[mon_idx])] > 45.0:
                print('JET POLEWARD OF 45: {} {}'.format(year, idx + 1))
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

                axis.plot(lat[:None], theta_xpv[idx, :None], 'k-')


                axis.plot(lat[:None], theta_cby)
                ax2.plot(lat[:None], dtheta_cby)
                ax2.plot(lat[jet_loc], dtheta_cby[jet_loc], 'o')
                #axis.plot(2 * [lat[dtheta_cby[1:-1].argmin() + 1]], axis.get_ylim(), '--')

                axis.plot(2 * [lat[int(jet_loc_ts[mon_idx])]], axis.get_ylim(), '--')
                axis.plot(lat[int(jet_loc_ts[mon_idx])],
                          theta_cby[int(jet_loc_ts[mon_idx])], 'ko', ms=5)

                ax2.set_ylim([-4, 4])
                ax2.grid(b=False)

                #plt.colorbar()
                axis.set_title('{:%Y-%m-%d %H}'.format(dates[idx]))
                plt.savefig('plt_pv_{}_{:04d}.png'.format(year, idx))
                plt.tight_layout()
                #plt.show()
                plt.close()

            mon_idx += 1

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(lat[jet_loc_ts.astype(int)], 'x-')
    plt.tight_layout()
    plt.savefig('plt_jet_loc_ts_{}-{}.pdf'.format(year_s, year_e))
    plt.show()
