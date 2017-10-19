#-*- coding: utf-8 -*-
"""
Module containing classes for diagnoistic variable calculation and diagnoistic plotting.
"""
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import basemap

import input_data
import stj_metric
import run_stj

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
        self.contours = None
        self.extn = 'eps'

        # Figure size set to 129 mm wide, 152 mm tall
        fig_mult = 1.0
        fig_width = 129 * fig_mult
        self.fig_size = (fig_width / 25.4, (fig_width / 0.9) / 25.4)

        plt.rc('text', usetex=True)
        plt.rc('text.latex', unicode=True)
        plt.rc('font', family='serif', size=8 * fig_mult)

    def test_method_plot(self, date):
        """
        Plot individual components of the metric to ensure it is working properly.

        Parmaeters
        ----------
        date : A :py:meth:`~datetime.datetime` instance to select and plot

        """
        data = input_data.InputData(self.props, date_s=date,
                                    date_e=date + dt.timedelta(seconds=3600 * 32 * 24))
        pv_lev = self.props.config['pv_value']
        data.get_data_input()
        self.stj = self.metric(self.props, data)
        tix = 0
        zix = 7
        fig = plt.figure(figsize=self.fig_size)
        axes = [plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2),
                plt.subplot2grid((3, 4), (0, 2), rowspan=2, colspan=2),
                plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=3),
                plt.subplot2grid((3, 4), (2, 3), rowspan=1, colspan=1)]

        # uwnd_max = np.max(np.abs(data.uwnd[tix, zix, ...]))
        # spc = uwnd_max // 13
        uwnd_max = 50.
        spc = 5.
        self.contours = np.arange(-np.ceil(uwnd_max), np.ceil(uwnd_max) + spc, spc)
        jet_lat = []

        for hidx, shem in enumerate([True, False]):
            dtheta, theta_fit, theta_xpv, select, lat, y_s, y_e = self._jet_details(shem)
            jet_pos = np.median(select[tix, :]).astype(int)
            jet_lat.append(lat[jet_pos])
            theta_fit_eval = self.stj.peval(lat[y_s:y_e], theta_fit)

            # Find the zonal mean zonal wind for this hemisphere
            uwnd_hemis = np.mean(data.uwnd[self.stj.hemis], axis=-1)[tix, ...]

            # Make contour plot (theta vs. lat) of zmzw
            axes[hidx].contourf(lat, data.th_lev, uwnd_hemis, self.contours,
                                cmap='RdBu_r', extend='both')

            # Plot mean theta profile
            axes[hidx].plot(lat, np.mean(theta_xpv[tix, ...], axis=-1), 'k.', ms=1.0,
                            label=r'$\theta_{%iPVU}$' % pv_lev)

            # Plot Mean theta fit
            axes[hidx].plot(lat[y_s:y_e], np.mean(theta_fit_eval[tix, ...], axis=0),
                            label=r'$\theta_{%iPVU}$ Fit' % pv_lev, lw=2.0)

            axes[hidx].plot(lat[jet_pos], np.mean(theta_xpv[tix, ...], axis=-1)[jet_pos],
                            'C0o', ms=5, label='Jet Location')

            # Restrict axis to only between 280 - 400K
            axes[hidx].set_ylim([300, 400])

            # Duplicate the axis
            ax2 = axes[hidx].twinx()
            # Plot meridional derivative of theta on X PVU
            ax2.plot(lat[y_s:y_e], np.mean(dtheta[tix, ...], axis=-1), 'C2',
                     label=r'$\partial\theta_{%iPVU}/\partial\phi$' % pv_lev, lw=2.0)

            # Restrict to +/- 4 so that both SH and NH have same Y-axis
            ax2.set_ylim([-4, 4])

            # Set the color to match the dTheta/dphi line
            ax2.tick_params('y', colors='C2')

            if shem:
                ax2.tick_params(right='off', labelright='off')
            else:
                axes[hidx].tick_params(left='off', labelleft='off')
            axes[hidx].tick_params(bottom='off', labelbottom='off',
                                   top='on', labeltop='on')
            if shem:
                lat_labels = np.arange(-90, 30, 30)
            else:
                lat_labels = np.arange(0, 90 + 30, 30)

            axes[hidx].set_xticks(lat_labels)
            axes[hidx].set_xticklabels([u'{}\u00B0'.format(lati) for lati in lat_labels],
                                       fontdict={'usetex': False})

            axes[hidx].set_xlabel(r'$\phi$')
            axes[hidx].grid(b=False)
            ax2.grid(b=False)

        # Plot wind map
        cfill = self.plot_uwnd(data, axes[2], (tix, zix), jet_lat)

        axes[0].set_ylabel(r'$\theta$ [K]')
        ax2.set_ylabel(r'$\partial\theta/\partial\phi$ [K/rad]', color='C2')

        # Combine legends from axes[1] and its twin
        h_1, l_1 = axes[1].get_legend_handles_labels()
        h_2, l_2 = ax2.get_legend_handles_labels()
        ax2.legend(h_1 + h_2, l_1 + l_2, loc='upper right', fancybox=False)

        # Add plot of zmzw as function of latitude
        axes[3].plot(np.mean(data.uwnd[tix, zix, ...], axis=-1), data.lat)
        for lati in jet_lat:
            axes[3].axhline(lati, color='k', lw=0.5)
        axes[3].set_ylim([-90, 90])

        axes[0].text(-90, 399, '(a)', verticalalignment='top', horizontalalignment='left')
        axes[1].text(0, 399, '(b)', verticalalignment='top', horizontalalignment='left')
        axes[2].set_title('(c)')
        axes[3].set_title('(d)')

        cbar_ax = fig.add_axes([0.15, 0.07, .7, .01])
        fig.colorbar(cfill, cax=cbar_ax, orientation='horizontal', format='%3.1f',
                     label=r'U-Wind [$m\,s^{-1}$]')

        fig.subplots_adjust(left=0.1, bottom=0.12, right=0.92, top=0.96,
                            wspace=0.03, hspace=0.27)
        axes[2].set_position([0.0, 0.12, 0.75, .24])
        plt.savefig('plt_jet_props_{}.{}'.format(date.strftime('%Y-%m'), self.extn))

    def plot_uwnd(self, data, axis, index, jet_lat):
        """
        Plot zonal wind for a specific time/level on a map.

        Parameters
        ----------
        data : :py:meth:`~STJ_PV.input_data.InputData`
            InputData object, contains u-wind, and PV field and coordinates
        axis : :py:meth:`matplotlib.pyplot.Axis`
            Axis on which to plot
        index : tuple
            Tuple of (time, level) indicies, respectively
        jet_lat : list
            Length 2 list of jet latitudes [NH, SH] or [SH, NH]

        Returns
        -------
        cfill : :py:meth:`~matplotlib.pyplot.contourf`
            Contour fill object from map

        """
        if data.lon[0] == 0 or data.lon[0] == 360.0:
            # If longitude is 0 - 360 then centre longitude is 180
            lon_0 = 180.0
        else:
            # Otherwise use 0 and hope for the best
            lon_0 = 0.0

        tix, zix = index
        pmap = basemap.Basemap(projection='eck4', lon_0=lon_0, resolution='c')
        # pmap = basemap.Basemap(projection='kav7', lon_0=lon_0, resolution='c')
        # pmap = basemap.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90,
        #                        llcrnrlon=0, urcrnrlon=360)
        uwnd, lon = basemap.addcyclic(data.uwnd[tix, zix, ...], data.lon)
        map_x, map_y = pmap(*np.meshgrid(lon, data.lat))

        cfill = pmap.contourf(map_x, map_y, uwnd, self.contours,
                              cmap='RdBu_r', ax=axis, extend='both')

        pmap.drawcoastlines(ax=axis, linewidth=0.5)
        lat_spc = 15.
        lon_spc = 30.
        line_props = {'ax': axis, 'linewidth': 0.4, 'dashes': [3, 3]}
        pmap.drawparallels(np.arange(-90, 90 + lat_spc, lat_spc), **line_props)
        pmap.drawmeridians(np.arange(0, 360 + lon_spc, lon_spc), **line_props)
        pmap.drawparallels(np.arange(-90, 90 + lat_spc * 2, lat_spc * 2),
                           labels=[True, False, False, False], **line_props)
        pmap.drawmeridians([30, 180, 330], labels=[False, False, False, True],
                           **line_props)

        # Draw horizontal lines for jet location in each hemisphere
        # Dashes list is [pixels on, pixels off], higher numbers are better
        # when using eps and trying to draw a solid line
        pmap.drawparallels(jet_lat, dashes=[5, 0], linewidth=1.5, ax=axis)

        return cfill


    def _jet_details(self, shemis=True):
        """Get Jet details using :py:meth:`~STJ_PV.stj_metric.STJPV` API for a hemisphere.
        """

        # --------------------- Code from STJMetric.find_jet() --------------------- #
        if shemis and self.stj.pv_lev < 0 or not shemis and self.stj.pv_lev > 0:
            pv_lev = np.array([self.stj.pv_lev])
        else:
            pv_lev = -1 * np.array([self.stj.pv_lev])

        lat, _, extrema = self.stj._set_hemis(shemis)
        theta_xpv, _, ushear = self.stj._isolate_pv(pv_lev)
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

def main():
    """Generate jet finder, make diagnostic plots."""

    jf_run = run_stj.JetFindRun('./conf/stj_config_ncep_monthly.yml')
    diags = DiagPlots(jf_run, stj_metric.STJPV)
    diags.test_method_plot(dt.datetime(2015, 1, 1))

    try:
        # Remove log file created by JF_RUN, comment this out if there's a problem
        os.remove(jf_run.config['log_file'])
    except OSError:
        print('Log file not found: {}'.format(jf_run.config['log_file']))
        pass


if __name__ == "__main__":
    main()
