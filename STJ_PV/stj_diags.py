# -*- coding: utf-8 -*-
"""
Module containing classes for diagnoistic variable calculation and diagnoistic plotting.
"""
import os
import datetime as dt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits import basemap

import input_data
import stj_metric
import run_stj

# Define the plot output extention
EXTN = 'pdf'


class DiagPlots(object):
    """
    Plot diagnostic metrics about subtropical jet properties.
    """

    def __init__(self, stj_props, metric, ilon=None):
        """
        Setup DiagMetrics, input from STJIPVMetric and STJProperties classes.
        """
        self.props = stj_props
        self.metric = metric
        self.stj = None
        self.contours = None
        self.jet_info = {'lat_all': [], 'jet_lat': [], 'jet_idx': [], 'jet_mean': []}
        self.ilon = ilon

        # Figure size set to 129 mm wide, 152 mm tall
        self.fig_mult = 2.

        plt.rc('text', usetex=True)
        plt.rc('text.latex', unicode=True)
        plt.rc('font', family='serif', size=8 * self.fig_mult)

    def _get_figsize(self, width=174):
        """Assign the figure size as function of width in mm."""
        fig_width = width * self.fig_mult
        return (fig_width / 25.4, (fig_width / 0.9) / 25.4)

    def test_method_plot(self, date):
        """
        Plot individual components of the metric to ensure it is working properly.

        Parmaeters
        ----------
        date : A :class:`datetime.datetime` instance to select and plot

        """
        data = input_data.InputDataSTJPV(self.props, date_s=date, date_e=date)
        data = data.get_data()
        vlat = self.props.data_cfg['lat']
        vlon = self.props.data_cfg['lon']

        if data[vlon].min() == 0.0:
            data = data.roll(**{vlon: data[vlon].shape[0] // 2})
            _lons = np.linspace(-180.0, 180.0, data[vlon].shape[0], endpoint=False)
            data = data.assign_coords(**{vlon: _lons})

        self.stj = self.metric(self.props, data)

        tix = 0
        zix = 6
        lat_idx = xr.DataArray(np.arange(data[vlat].shape[0]),
                               dims=[vlat], coords={vlat: data[vlat]})

        fig = plt.figure(figsize=self._get_figsize(width=129))
        axes = [plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2),
                plt.subplot2grid((3, 4), (0, 2), rowspan=2, colspan=2),
                plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=3),
                plt.subplot2grid((3, 4), (2, 3), rowspan=1, colspan=1)]

        uwnd_max = 50.
        spc = 5.
        self.contours = np.arange(-np.ceil(uwnd_max), np.ceil(uwnd_max) + spc, spc)

        for hidx, shem in enumerate([True, False]):
            ax2 = self.plot_zonal_slice(data, shem, hidx, tix, axes)

        axes[0].set_ylabel(r'$\theta$ [K]', rotation='horizontal')
        ax2.set_ylabel(r'$\partial\theta/\partial\phi$ [K/rad]', color='C2')

        # Combine legends from axes[1] and its twin
        h_1, l_1 = axes[1].get_legend_handles_labels()
        h_2, l_2 = ax2.get_legend_handles_labels()
        ax2.legend(h_1 + h_2, l_1 + l_2, loc='upper right', fancybox=False)

        # Plot wind map
        # If the zonal_opt is not 'mean', then plot the jet lat at each longitude
        zm_opt = self.props.config['zonal_opt'] == 'mean'
        cfill, pmap = self.plot_uwnd(data, axes[2], (tix, zix), map_zm=zm_opt)

        # Add plot of zmzw as function of latitude
        axes[3].spines['right'].set_color('none')
        axes[3].spines['top'].set_color('none')
        _, map_y = pmap(*np.meshgrid(data[vlon], data[vlat]))

        axes[3].plot(np.mean(data.uwnd[tix, zix, ...], axis=-1), map_y[:, 0],
                     '#fc4f30', lw=1.5 * self.fig_mult)

        for hidx, jet_pos in enumerate(self.jet_info['jet_mean']):
            lati = lat_idx.sel(**{vlat: jet_pos}, method='nearest').values
            axes[3].axhline(map_y[lati, 0], color='k', lw=1.8 * self.fig_mult)
            # Mean vs. median
            # lat_median = np.ma.mean(self.jet_info['lat_all'][hidx][tix])
            # lat_iqr = np.ma.std(self.jet_info['lat_all'][hidx][tix])

            # _, lat_q1 = pmap(0, lat_median - lat_iqr)
            # _, lat_q2 = pmap(0, lat_median + lat_iqr)

            axes[3].axhline(map_y[lati, 0], color='k', lw=1.8 * self.fig_mult)
            # axes[3].axhline(lat_q1, color='k', ls='--', lw=0.8 * self.fig_mult)
            # axes[3].axhline(lat_q2, color='k', ls='--', lw=0.8 * self.fig_mult)

            x_loc = axes[3].get_xlim()[-1]
            y_loc = map_y[lati, 0] * 1.03
            axes[3].text(x_loc, y_loc, '{:.1f}'.format(data[vlat][lati].values),
                         verticalalignment='bottom', horizontalalignment='left')

        axes[3].tick_params(left=True, labelleft=True, labeltop=False, right=False,
                            labelright=False)
        axes[3].ticklabel_format(axis='y', style='plain')
        ytick_lats = np.arange(-60, 60 + 30, 30)
        _, yticks = pmap(np.zeros(ytick_lats.shape), ytick_lats)
        axes[3].set_yticks(yticks)
        axes[3].set_yticklabels(ytick_lats)
        axes[3].set_ylim([map_y[:, 0].min(), map_y[:, 0].max()])

        axes[0].text(-88, 399, '(a)', verticalalignment='top',
                     horizontalalignment='left', fontsize=12 * self.fig_mult)
        axes[1].text(2, 399, '(b)', verticalalignment='top',
                     horizontalalignment='left', fontsize=12 * self.fig_mult)
        axes[2].set_title('(c)')
        axes[3].set_title('(d)')
        axes[3].set_xlabel(r'u wind [$m\,s^{-1}$]')

        # Colorbar axis
        cbar_ax = fig.add_axes([0.06, 0.25, .01, .5])
        cbar = fig.colorbar(cfill, cax=cbar_ax, orientation='vertical', format='%.0f',
                            label=r'u wind [$m\,s^{-1}$]')

        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation='vertical')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        fig.subplots_adjust(left=0.15, bottom=0.10, right=0.92, top=0.96,
                            wspace=0.0, hspace=0.27)

        # Basemap is finicky, so subplots_adjust doesn't always work with it well
        # Use this to set the position of the map so it lines up with the zonal mean
        # diagram, Axis.get_position() returns the location of each axis, the .bounds
        # of this gives the [x, y, width, height], which matches the Axis.set_position()
        axes[2].set_position([0.0, 0.12, 0.75, .24])
        ax3_c = axes[3].get_position().bounds
        ax2_c = axes[2].get_position().bounds
        axes[2].set_position([ax2_c[0] + 0.03, ax3_c[1], ax2_c[2], ax2_c[3]])
        out_file = ('plt_stj_diag_{}_{:.0f}K_{}'
                    .format(self.props.data_cfg['short_name'],
                            data['level'][zix].values, date.strftime('%Y-%m-%d')))
        out_file = out_file.replace('.', 'p')
        if self.ilon is None:
            plt.savefig('{}.{}'.format(out_file, EXTN))
        else:
            plt.savefig('{}_{}.{}'.format(out_file, self.ilon, EXTN))

        plt.clf()
        plt.close()

    def plot_zonal_slice(self, data, shem, hidx, tix, axes):
        """
        Plots zonal mean jet diagnostic picture.
        """
        lwid = 2.0 * self.fig_mult
        vlat = self.props.data_cfg['lat']
        vlev = self.props.data_cfg['lev']
        vlon = self.props.data_cfg['lon']
        dtheta, theta_fit, theta_xpv, select, lat, y_s, y_e = self._jet_details(shem)
        if self.ilon is None:
            jet_pos = select[tix, :].mean(dim=vlon)
        else:
            jet_pos = select[tix, :].sel(**{vlon: self.ilon}, method="nearest")


        # Append the list of jet latitude at each longitude in this hemisphere
        self.jet_info['lat_all'].append(select[tix])
        # Append the median jet latitude in this hemisphere to the list
        self.jet_info['jet_lat'].append(jet_pos)
        # Always keep the zonal mean jet position too
        self.jet_info['jet_mean'].append(select[tix, :].mean(dim=vlon))
        # Append the index of this hemisphere's jet on the full set of latitudes
        # self.jet_info['jet_idx'].append(list(data.lat).index(lat[jet_pos]))

        # Evaluate the polynomial fit of the pv surface for plotting
        theta_fit_eval = self.stj.peval(lat.values, theta_fit.values)
        theta_fit_eval = xr.DataArray(theta_fit_eval, dims=('time', vlon, vlat),
                                      coords={dimn: theta_xpv[dimn]
                                              for dimn in ['time', vlon, vlat]})

        # Find the zonal mean zonal wind for this hemisphere
        if self.ilon is None:
            uwnd_hemis = data.uwnd.sel(**self.stj.hemis).mean(dim=vlon)[tix]
        else:
            uwnd_hemis = (data.uwnd.sel(**self.stj.hemis)
                          .sel(**{vlon: self.ilon}, method="nearest")[tix])

        # Make contour plot (theta vs. lat) of zmzw
        axes[hidx].contourf(uwnd_hemis[vlat], uwnd_hemis[vlev],
                            uwnd_hemis, self.contours, cmap='RdBu_r', extend='both')

        # Plot mean theta profile
        if self.ilon is None:
            theta_mean_zm = theta_xpv[tix].mean(dim=vlon)
        else:
            theta_mean_zm = theta_xpv[tix].sel(**{vlon: self.ilon}, method="nearest")

        axes[hidx].plot(theta_xpv[vlat], theta_mean_zm, 'k.',
                        ms=2.0 * self.fig_mult,
                        label=r'$\theta_{%i}$' % self.props.config['pv_value'])

        # Plot Mean theta fit
        if self.ilon is None:
            theta_fit_zm = theta_fit_eval.mean(dim=vlon)
        else:
            theta_fit_zm = theta_fit_eval.sel(**{vlon: self.ilon}, method="nearest")

        axes[hidx].plot(theta_fit_zm[vlat], theta_fit_zm.squeeze(), 'C0-',
                        label=r'$\theta_{%i}$ Fit' % self.props.config['pv_value'],
                        lw=lwid)

        axes[hidx].plot(jet_pos, theta_fit_zm.sel(**{vlat: jet_pos.drop('time')},
                                                  method='nearest'),
                        'C0o', ms=5 * self.fig_mult, label='Jet')

        # Duplicate the axis
        ax2 = axes[hidx].twinx()
        if self.ilon is None:
            dtheta_zm = dtheta[tix].mean(dim=vlon)
        else:
            dtheta_zm = dtheta[tix].sel(**{vlon: self.ilon}, method="nearest")

        ax2.plot(dtheta[vlat], dtheta_zm,
                 'C2--', label=r'$\partial\theta_{%i}/\partial\phi$' %
                 self.props.config['pv_value'], lw=lwid)

        # Restrict axis to only between 280 - 400K
        axes[hidx].set_ylim([300, 400])

        if self.ilon is None:
            # Restrict to +/- 4 so that both SH and NH have same Y-axis
            ax2.set_ylim([-4, 4])
        else:
            ax2.set_ylim([-4, 4])

        # Set the color to match the dTheta/dphi line
        ax2.tick_params('y', colors='C2')

        # Set hemisphere specific plotting parameters
        if hidx == 0:
            axes[hidx].spines['right'].set_color('none')
            ax2.spines['right'].set_color('none')
            ax2.tick_params(right=False, labelright=False)
            lat_labels = np.arange(-90, 30, 30)
        else:
            axes[hidx].spines['left'].set_color('none')
            ax2.spines['left'].set_color('none')
            axes[hidx].tick_params(left=False, labelleft=False)
            lat_labels = np.arange(30, 90 + 30, 30)

        axes[hidx].tick_params(bottom=False, labelbottom=False,
                               top=True, labeltop=True)

        axes[hidx].set_xticks(lat_labels)
        axes[hidx].set_xticklabels([u'{}\u00B0'.format(lati) for lati in lat_labels],
                                   fontdict={'usetex': True})

        axes[hidx].grid(b=False)
        # Set the axis limits for hemisphere correctly
        for axis in [axes[hidx], ax2]:
            if hidx == 0:
                lims = [-90, 0]
            else:
                lims = [0, 90]
            axis.set_xlim(lims)

        ax2.grid(b=False)
        print('Jet at: {:.1f}'.format(jet_pos.values))
        # import pdb;pdb.set_trace()
        return ax2

    def plot_uwnd(self, data, axis, index, map_zm=True):
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

        Returns
        -------
        cfill : :py:meth:`~matplotlib.pyplot.contourf`
            Contour fill object from map

        """
        vlon = self.props.data_cfg['lon']
        vlat = self.props.data_cfg['lat']

        if data[vlon][0] == 0 or data[vlon][0] == 360.0:
            # If longitude is 0 - 360 then centre longitude is 180
            lon_0 = 180.0
        else:
            # Otherwise use 0 and hope for the best
            lon_0 = 0.0

        tix, zix = index
        pmap = basemap.Basemap(projection='eck4', lon_0=lon_0, resolution='c', ax=axis)
        # pmap = basemap.Basemap(projection='kav7', lon_0=lon_0, resolution='c')
        # pmap = basemap.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90,
        #                        llcrnrlon=0, urcrnrlon=360)

        uwnd = data.uwnd[tix, zix, ...].values
        lon = data[vlon].values
        lat = data[vlat]

        map_x, map_y = pmap(*np.meshgrid(lon, lat))

        cfill = pmap.contourf(map_x, map_y, uwnd, self.contours,
                              cmap='RdBu_r', ax=axis, extend='both')

        pmap.drawcoastlines(ax=axis, linewidth=0.5, color='grey')
        lat_spc = 15
        lon_spc = 30
        line_props = {'ax': axis, 'linewidth': 0.1, 'dashes': [3, 10]}
        _lat_grid = np.arange(-90, 90 + lat_spc, lat_spc)
        _lon_grid = np.arange(0, 360 + lon_spc, lon_spc)
        pmap.drawparallels(_lat_grid, **line_props)
        pmap.drawmeridians(_lon_grid, **line_props)

        pmap.drawparallels(
            _lat_grid[::4], labels=[True, False, False, False], **line_props
        )
        pmap.drawmeridians(
            _lon_grid[::4], labels=[False, False, False, True], **line_props
        )

        # Draw horizontal lines for jet location in each hemisphere
        # Dashes list is [pixels on, pixels off], higher numbers are better
        # when using eps and trying to draw a solid line
        if map_zm:
            pmap.drawparallels(self.jet_info['jet_lat'], dashes=[5, 0],
                               linewidth=1.5, ax=axis)
        else:
            for hidx in [0, 1]:
                if len(self.jet_info['lat_all'][hidx]) == data[vlon].shape[0]:
                    xpt, ypt = pmap(data[vlon].values,
                                    self.jet_info['lat_all'][hidx].values)
                else:
                    xpt, ypt = pmap(data[vlon].values,
                                    self.jet_info['lat_all'][hidx][tix].values)
                pmap.plot(xpt, ypt, 'ko', ms=1.8, ax=axis)

        return cfill, pmap

    def _jet_details(self, shemis=True):
        """Get Jet details using :py:meth:`~STJ_PV.stj_metric.STJPV` API for a hemisphere.
        """
        dtheta, theta_fit, theta_xpv, select = self.stj.find_jet(shemis, debug=True)
        lat = theta_xpv[self.props.data_cfg['lat']]
        y_s = None
        y_e = None

        # --------------------- Code from STJMetric.find_jet() --------------------- #
        return dtheta, theta_fit, theta_xpv, select, lat, y_s, y_e


def main():
    """Generate jet finder, make diagnostic plots."""

    dates = [dt.datetime(2013, 1, 1), dt.datetime(2013, 6, 1)]
    # dates = pd.date_range('1981-02-05', '1981-02-20', freq='D')

    # This loop does not work well if outputting to .eps files, just run the code twice
    for date in dates:
        # jf_run = run_stj.JetFindRun('./conf/stj_config_erai_monthly_gv.yml')
        # jf_run = run_stj.JetFindRun('./conf/stj_config_ncep_monthly.yml')
        # jf_run = run_stj.JetFindRun('./conf/stj_config_erai_monthly.yml')
        # jf_run = run_stj.JetFindRun('./conf/stj_config_ncep.yml')
        # jf_run =  run_stj.JetFindRun('./conf/stj_config_jra55_theta_mon.yml')
        jf_run = run_stj.JetFindRun('./conf/stj_config_erai_theta.yml')

        # Force update_pv and force_write to be False, optional override of zonal-mean
        jf_run.config['update_pv'] = False
        jf_run.config['force_write'] = False
        jf_run.config['zonal_opt'] = 'indv'
        diags = DiagPlots(jf_run, stj_metric.STJPV, ilon=180.)
        diags.test_method_plot(date)

        try:
            # Remove log file created by JF_RUN, comment this out if there's a problem
            os.remove(jf_run.config['log_file'])
        except OSError:
            print('Log file not found: {}'.format(jf_run.config['log_file']))


if __name__ == "__main__":
    main()
