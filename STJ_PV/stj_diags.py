#-*- coding: utf-8 -*-
"""
Module containing classes for diagnoistic variable calculation and diagnoistic plotting.
"""
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import basemap

import input_data
import stj_metric
import run_stj


class DiagMetrics(object):
    """
    Calculate diagnostic metrics about subtropical jet properties.
    """

    def __init__(self, stj_props, metric):
        """
        Setup DiagMetrics, input from a STJMetric and STJProperties classes.
        """
        self.props = stj_props
        self.metric = metric

    def jet_intensity(self):
        """
        Calculate jet intensity at identified points.
        """

    def annual_correlations(self):
        """
        Calculate annual correlations.
        """

    def monthly_correlations(self):
        """
        Calculate monthly correlations.
        """

    def seasonal_correlations(self):
        """
        Calculate seasonal correlations.
        """

    def polyfit_near_mean(self):
        """
        Perform a polynomial fit of pv contour near "expected" jet location mean.
        """

    def validate_near_mean(self):
        """
        Check that identified STJ position is near "expected" jet location mean.
        """

    def get_uwind_strength(self):
        """
        Calculate zonal wind profile.
        """
        # created named tuples to manage storage of index
        metric = collections.namedtuple('metric', 'name hemisphere intensity position')
        time_len = self.ipv_data['ipv'].shape[0]
        STJ_int = np.zeros(time_len)
        STJ_pos = np.zeros(time_len)
        pdb.set_trace()

        for hemi in ['NH', 'SH']:
            for time_loop in range(time_len):
                # for time_loop in xrange(1):

                if hemi == 'NH':
                    lat = self.lat_NH
                    STJ_phi = self.NH_STJ_phi[time_loop]
                    STJ_th = self.NH_STJ_theta[time_loop]
                else:
                    lat = self.lat_SH
                    STJ_phi = self.SH_STJ_phi[time_loop]
                    STJ_th = self.SH_STJ_theta[time_loop]

                # step 8. interpolate u wind
                u_zonal = MeanOverDim(data=self.u_th[time_loop, :, :, :], dim=2)
                u_zonal_function = interpolate.interp2d(
                    self.lat, self.theta_lev, u_zonal, kind='cubic')
                u_zonal_interp = u_zonal_function(lat, self.theta_interp)

                # step 9: for the 2.0 max derivative latitude find the uwind strength
                # get element closest to phi and theta points

                elem_phi = FindClosestElem(STJ_phi, lat)[0]
                elem_theta = FindClosestElem(STJ_th, self.theta_interp)[0]

                STJ_int[time_loop] = u_zonal_interp[elem_theta, elem_phi]
                STJ_pos[time_loop] = STJ_phi

            if hemi == 'NH':
                Metric_NH = metric(name='STJ', hemisphere=hemi,
                                   intensity=STJ_int, position=STJ_pos)
            else:
                Metric_SH = metric(name='STJ', hemisphere=hemi,
                                   intensity=STJ_int, position=STJ_pos)

        return Metric_NH, Metric_SH

class DiagPlots(object):
    """
    Plot diagnostic metrics about subtropical jet properties.
    """

    def __init__(self, stj_props, metric):
        """
        Setup DiagMetrics, input from STJIPVMetric and STJProperties classes.
        """
        self.props = stj_props
        self.metric = metric
        self.stj = None

    def test_method_plot(self, date):
        """
        Plot individual components of the metric to ensure it is working properly.

        Parmaeters
        ----------
        date : A :py:meth:`~datetime.datetime` instance to select and plot

        """
        data = input_data.InputData(self.props, date_s=date,
                                    date_e=date + dt.timedelta(seconds=3600 * 32 * 24))
        data.get_data_input()
        self.stj = self.metric(self.props, data)
        tix = 0
        zix = 4

        fig = plt.figure(figsize=(9, 10))
        axes = [plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2),
                plt.subplot2grid((3, 4), (0, 2), rowspan=2, colspan=2),
                plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=3),
                plt.subplot2grid((3, 4), (2, 3), rowspan=1, colspan=1)]

        cfill, contours = self.plot_uwnd(data, axes[2], (tix, zix))
        for hidx, shem in enumerate([True, False]):
            dtheta, theta_fit, theta_xpv, select, lat, y_s, y_e = self._jet_details(shem)
            jet_pos = np.median(select[tix, :]).astype(int)

            theta_fit_eval = self.stj.peval(lat[y_s:y_e], theta_fit)
            # Find the zonal mean zonal wind for this hemisphere
            uwnd_hemis = np.mean(data.uwnd[self.stj.hemis], axis=-1)[tix, ...]

            # Make contour plot (theta vs. lat) of zmzw
            axes[hidx].contourf(lat, data.th_lev, uwnd_hemis, contours, cmap='RdBu_r')
            # Plot Mean theta fit
            axes[hidx].plot(lat[y_s:y_e], np.mean(theta_fit_eval[tix, ...], axis=0))
            # Plot mean theta profile
            axes[hidx].plot(lat, np.mean(theta_xpv[tix, ...], axis=-1))
            axes[hidx].plot(lat[jet_pos], np.mean(theta_xpv[tix, ...], axis=-1)[jet_pos],
                            'C0o', ms=5)

            # Restrict axis to only between 280 - 400K
            axes[hidx].set_ylim([280, 400])

            # Duplicate the axis
            ax2 = axes[hidx].twinx()
            # Plot meridional derivative of theta on X PVU
            ax2.plot(lat[y_s:y_e], np.mean(dtheta[tix, ...], axis=-1), 'C2')
            # Restrict to +/- 4 so that both SH and NH have same Y-axis
            ax2.set_ylim([-4, 4])
            if shem:
                ax2.tick_params(right='off', labelright='off')
            else:
                axes[hidx].tick_params(left='off', labelleft='off')

        # Add plot of zmzw as function of latitude
        axes[3].plot(np.mean(data.uwnd[tix, zix, ...], axis=-1), data.lat)
        plt.tight_layout()
        plt.show()

    def plot_uwnd(self, data, axis, index):
        if data.lon[0] == 0 or data.lon[0] == 360.0:
            # If longitude is 0 - 360 then centre longitude is 180
            lon_0 = 180.0
        else:
            # Otherwise use 0 and hope for the best
            lon_0 = 0.0

        tix, zix = index
        pmap = basemap.Basemap(projection='kav7', lon_0=lon_0, resolution='c')
        map_x, map_y = pmap(*np.meshgrid(data.lon, data.lat))
        uwnd_max = np.max(np.abs(data.uwnd[tix, zix, ...]))
        contours = np.linspace(-np.ceil(uwnd_max), np.ceil(uwnd_max), 15)
        cfill = pmap.contourf(map_x, map_y, data.uwnd[tix, zix, ...], contours,
                              cmap='RdBu_r', ax=axis)
        pmap.drawcoastlines(ax=axis)
        lat_spc = 15.
        lon_spc = 30.
        pmap.drawparallels(np.arange(-90, 90 + lat_spc, lat_spc), ax=axis)
        pmap.drawmeridians(np.arange(0, 360 + lon_spc, lon_spc), ax=axis)

        return cfill, contours


    def _jet_details(self, shemis=True):

        # --------------------- Code from STJMetric.find_jet() --------------------- #
        if shemis and self.stj.pv_lev < 0 or not shemis and self.stj.pv_lev > 0:
            pv_lev = np.array([self.stj.pv_lev])
        else:
            pv_lev = -1 * np.array([self.stj.pv_lev])

        lat, hidx, extrema = self.stj._set_hemis(shemis)
        theta_xpv, uwnd_xpv, ushear = self.stj._isolate_pv(pv_lev)
        dims = theta_xpv.shape


        # ----------------- Code from STJMetric._find_single_jet() ----------------- #
        # Restrict interpolation domain to a "reasonable" subset using a minimum latitude
        y_s = np.abs(np.abs(lat) - self.props.config['min_lat']).argmin()
        y_e = None

        # If latitude is in decreasing order, switch start & end
        # This makes sure we're selecting the latitude nearest the equator
        if abs(lat[0]) > abs(lat[-1]):
            y_s, y_e = y_e, y_s
        dims_fit = theta_xpv[:, y_s:y_e, :].shape
        tht_fit_shape = (self.props.config['fit_deg'] + 1, dims_fit[0], dims_fit[-1])

        dtheta = np.zeros(dims_fit)
        theta_fit = np.zeros(tht_fit_shape)
        select = np.zeros((dims_fit[0], dims_fit[-1]))

        for tix in range(dims[0]):
            for xix in range(dims[-1]):
                # Find derivative of dynamical tropopause
                jet_info = self.stj._find_single_jet(theta_xpv[tix, :, xix], lat,
                                                     ushear[tix, :, xix], extrema,
                                                     debug=True)

                select[tix, xix] = jet_info[0]

                dtheta[tix, :, xix] = jet_info[2]
                theta_fit[:, tix, xix] = jet_info[3][0]
                y_s, y_e = jet_info[-2:]

        return dtheta, theta_fit, theta_xpv, select, lat, y_s, y_e

    def compare_fd_cby(self):
        """
        Plot comparison of finite difference and Chebyshev polynomial derivative jet pos.
        """

    def test_second_derr(self):
        """
        Plot test of second derivative of PV contour.
        """

    def jet_lat_timeseries(self):
        """
        Plot jet location over time for both NH and SH.
        """

    def partial_correlation_matrix(self):
        """
        Plot matrix of partial correlations.
        """

if __name__ == "__main__":
    jf_run = run_stj.JetFindRun('./conf/stj_config_ncep_monthly.yml')
    diags = DiagPlots(jf_run, stj_metric.STJPV)
    diags.test_method_plot(dt.datetime(2012, 6, 1))
