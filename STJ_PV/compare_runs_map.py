#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare two STJ metrics. Plot limited timeseries and a map."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns
import compare_two_runs as c2r
from mpl_toolkits import basemap as bmp
import run_stj
__author__ = 'Michael Kelleher'

HEM_LIMS = {'nh': [23, 54], 'sh': [-54, -23]}
INFO = {'ERAI-PVGrad':
        {'file': ('ERAI_MONTHLY_THETA_STJPV_pv2.0_fit6_y010.0_yN65.0'
                  '_zmean_1979-01-01_2016-12-31.nc'), 'label': 'PV Grad'},

        'ERAI-Uwind':
        {'file': ('ERAI_PRES_STJUMax_pres25000.0_y010.0_yN65.0_zmean_'
                  '1979-01-01_2016-12-31.nc'), 'label': 'U Max'}}


def draw_map_lines(pmap, axis):
    """
    Draw lat/lon and coast lines on a Basemap.

    Parameters
    ----------
    pmap : :class:`mpl_toolkits.basemap.Basemap`
        A `Basemap` onto which lines will be drawn
    axis : :class:`matplotlib.axes.Axes`
        Axis reference for `pmap`

    Returns
    -------
    None

    """
    lat_spc = 30
    lon_spc = 60
    line_props = {'ax': axis, 'linewidth': 0.2, 'dashes': [3, 10]}
    pmap.drawparallels(np.arange(-90, 90 + lat_spc, lat_spc), **line_props)
    pmap.drawmeridians(np.arange(0, 360 + lon_spc, lon_spc), **line_props)
    pmap.drawparallels(np.arange(-90, 90 + lat_spc * 2, lat_spc * 2),
                       labels=[True, False, False, False], **line_props)
    pmap.drawcoastlines(linewidth=line_props['linewidth'], ax=axis)
    circle = pmap.drawmapboundary(linewidth=2, color='white', ax=axis,
                                  zorder=6, fill_color='none')
    circle.set_clip_on(False)


def get_pvgrad_pos(date):
    """
    Get STJ position at all longitudes for PVGrad method.

    Parameters
    ----------
    date : :class:`datetime.datetime`
        Selected date to compute STJ metric

    Returns
    -------
    jet_lat : :class:`numpy.ndarray`
        An array of jet latitude locations (hemisphere, time, lon)

    """
    jf_run = run_stj.JetFindRun('./conf/stj_config_erai_theta.yml')
    # Force update_pv and force_write to be False,
    # optional override of zonal-mean
    jf_run.config['update_pv'] = False
    jf_run.config['force_write'] = False
    jf_run.config['zonal_opt'] = 'indv'
    jet = jf_run.run(date_s=date, date_e=date + pd.Timedelta(days=34),
                     save=False)
    try:
        # Remove log file created by JF_RUN, comment
        # this out if there's a problem
        os.remove(jf_run.config['log_file'])
    except OSError:
        print('Log file not found: {}'.format(jf_run.config['log_file']))

    return jet.jet_lat


def plot_annotations(fig, axes, cfill):
    """Annotate the map and line plots.
    """
    grid_style = {'b': True, 'ls': '-', 'color': 'lightgrey', 'lw': 0.5}
    axes[0, 0].legend(ncol=2, fontsize=plt.rcParams['font.size'])
    axes[0, 0].tick_params(bottom='off', labelbottom='off')
    for idx in range(axes.shape[0]):

        # Remove borders from timeseries plot
        sns.despine(ax=axes[idx, 0], left=False, bottom=idx == 0, offset=4)
        # Set the lineplot grid to have `grid_style`
        axes[idx, 0].grid(**grid_style)

    # Add colorbar axis
    cax = fig.add_axes([0.5, 0.04, 0.45, 0.015])
    cbar = fig.colorbar(cfill, cax=cax, orientation='horizontal')
    # cbar.ax.yaxis.set_ticks_position('right')
    # cbar.ax.yaxis.set_label_position('right')

    # Make colorbar border thinner
    cbar.outline.set_linewidth(0.1)

    fig.subplots_adjust(left=0.07, bottom=0.05,
                        right=0.94, top=0.98,
                        wspace=0.03, hspace=0.03)


def main(width=174, figscale=1.0, extn='png'):
    """Load data, make plots."""
    # Parameters, labels, etc.
    in_names = ['ERAI-PVGrad', 'ERAI-Uwind']
    labels = [INFO[name]['label'] for name in in_names]
    dates = {'nh': pd.Timestamp('2010-07-01'),
             'sh': pd.Timestamp('2016-03-01')}
    nc_dir = './jet_out'
    wind_dir = '/Volumes/FN_2187/erai/monthly/'
    theta_lev = 350

    # Set the font to sans-serif and size 9 (but scaled)
    plt.rc('font', family='sans-serif', size=9 * figscale)
    # Adjust the title padding to bring it closer
    # this won't work under axis.set_title???
    plt.rc('axes', titlepad=-5.0)

    # Assign colors to labels so lines on timeseries and map are the same color
    cols = {labels[0]: '#0D1317', labels[1]: '#5755BA'}

    # Set contour levels for map,
    u_contours = np.arange(-45, 50, 5)
    # Load file diags using compare_two_runs.FileDiag and append two metrics
    fds = [c2r.FileDiag(INFO[in_name], file_path=nc_dir)
           for in_name in in_names]
    data = fds[0].append_metric(fds[1])

    # Make timeseries plot for each hemisphere, and map for selected date
    # Scale the figure size
    fig_w, fig_h = (figscale * width / 25.4, figscale * width / 25.4)

    # Make a 2 x 2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    for idx, dfh in enumerate(data.groupby('hem')):
        # Hemisphere key-name
        hem = dfh[0]

        # Plot timeseries for each method
        for kind, dfk in dfh[1].groupby('kind'):
            axes[idx, 0].plot(dfk.lat, label=kind, color=cols[kind], lw=1.5)

        # Label the timeseries
        axes[idx, 0].set_ylabel(c2r.HEMS[hem])

        # Restrict to 2010-2016
        axes[idx, 0].set_xlim([pd.Timestamp('2010-01-01'),
                               pd.Timestamp('2016-06-30')])

        # Show which date is being plotted in the map with a verical line
        axes[idx, 0].axvline(dates[hem], color='k', ls='--', lw=1.1)

        # Rotate y-axis ticks so SH/NH have same
        # left-ward extent, limit the y-axis
        axes[idx, 0].tick_params(axis='y', rotation=90)
        axes[idx, 0].set_ylim(HEM_LIMS[hem])

        # Open ERAI data to extract zonal wind
        dsw = xr.open_dataset(f'{wind_dir}/erai_theta_{dates[hem].year}.nc')
        # Select the correct day and level from the u-wind
        uwnd = dsw.sel(time=dates[hem], level=theta_lev).u

        # Get latitude and add cyclic point from longitude
        lat = uwnd.latitude.values
        uwnd, lon = bmp.addcyclic(uwnd.values, uwnd.longitude.values)

        # Generate a {s/n}pstere Basemap
        pmap = bmp.Basemap(projection=f'{hem[0]}pstere',
                           lon_0=0, boundinglat=0,
                           resolution='c', round=True)

        map_x, map_y = pmap(*np.meshgrid(lon, lat))
        cfill = pmap.contourf(map_x, map_y, uwnd, u_contours,
                              cmap='RdBu_r', ax=axes[idx, 1], extend='both')

        # Extract the kind for the zonal mean of the first kind (labels[0])
        _umax = dfh[1][dfh[1].kind == labels[1]]
        # Generate 40 longitude, 40 latitude points from
        # the identified zonal mean position for the particular
        # day, transform them to the `pmap` coordinates
        umax_map = pmap(np.linspace(0, 360, 40),
                        [_umax[_umax.time == dates[hem]].lat[0]] * 40)

        pmap.plot(*umax_map, color=cols[labels[1]], ax=axes[idx, 1],
                  linewidth=1.5, zorder=5)

        # Create and run an stj_metric.STJPVMetric, don't save,
        # just return lat position
        pv_grad_lat = get_pvgrad_pos(dates[hem])

        # Indicies in this array are opposite to this loop's `idx`
        if hem == 'sh':
            hem_idx = 0
        else:
            hem_idx = 1

        # Transform the longitude and latitude points of the identified jet
        # to map coords then plot it on pmap
        pvgrad_map = pmap(lon[:-1], pv_grad_lat[hem_idx, 0])
        pmap.plot(*pvgrad_map, 'o', color=cols[labels[0]],
                  ms=0.6, ax=axes[idx, 1])

        # Label the map with the selected date, align the title to
        # the right, the padding is set by plt.rcParams['axes.titlepad'],
        # rather than hlightgreyere, for some reason
        axes[idx, 1].set_title(dates[hem].strftime('%b %Y'), loc='right')
        draw_map_lines(pmap, axes[idx, 1])

    plot_annotations(fig, axes, cfill)
    plt.savefig('plt_diag_ts_map_{}-{}.{ext}'.format(ext=extn, *in_names))


if __name__ == '__main__':
    main(extn='pdf')
