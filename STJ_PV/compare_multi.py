#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare multiple STJ.JetRun outputs on one figure."""
import numpy as np
import matplotlib.pyplot as plt
from compare_two_runs import FileDiag
import seaborn as sns

__author__ = 'Michael Kelleher'


def main(extn='pdf', fig_mult=1.0):
    """Load and combine multiple jet runs into one DataFrame, plot info."""
    data = {'ERAI-Monthly':
            {'file': ('ERAI_MONTHLY_THETA_STJPV_pv2.0_fit6_y010.0_yN65.0'
                      '_zmean_1979-01-01_2016-12-31.nc'),
             'label': 'Monthly ERA-I'},

            'ERAI-Daily':
            {'file': ('ERAI_DAILY_THETA_STJPV_pv2.0_fit6_y010.0_yN65.0'
                      '_zmean_1979-01-01_2017-12-31.nc'),
             'label': 'Daily ERA-I'},

            'MERRA-Monthly':
            {'file': ('MERRA_MONTHLY_STJPV_pv2.0_fit6_y010.0_yN65.0'
                      '_zmean_1980-01-01_2017-12-31.nc'),
             'label': 'Monthly MERRA'},

            'MERRA-Daily':
            {'file': ('MERRA_DAILY_STJPV_pv2.0_fit6_y010.0_yN65.0'
                      '_zmean_1980-01-01_2017-12-31.nc'),
             'label': 'Daily MERRA'},

            'NCEP-Monthly':
            {'file': ('NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit6_y010.0_yN65.0'
                      '_zmean_1979-01-01_2016-12-31.nc'),
             'label': 'Monthly NCEP'},

            'NCEP-Daily':
            {'file': ('NCEP_NCAR_DAILY_STJPV_pv2.0_fit6_y010.0_yN65.0'
                      '_zmean_1979-01-01_2016-12-31.nc'),
             'label': 'Daily NCEP'}}

    dsets = ['ERAI-Daily', 'ERAI-Monthly', 'MERRA-Daily', 'MERRA-Monthly',
             'NCEP-Daily', 'NCEP-Monthly']

    fds = [FileDiag(data[dset], file_path='jet_out') for dset in dsets]
    metric = fds[0].metric
    for fdi in fds[1:]:
        metric = metric.append(fdi.metric)

    plt.rc('font', size=8 * fig_mult)

    sns.set_style('whitegrid')
    fig_width = (12.9 / 2.54) * fig_mult
    fig_height = fig_width * 0.6

    hems = {'nh': {'ticks': np.arange(20, 60, 10),
                   'min_ticks': np.arange(20, 55, 5),
                   'ylims': (15, 50),
                   'sea_order': ['DJF', 'MAM', 'JJA', 'SON']},
            'sh': {'ticks': np.arange(-10, -55, -10),
                   'min_ticks': np.arange(-10, -55, -5),
                   'ylims': (-15, -50),
                   'sea_order': ['JJA', 'SON', 'DJF', 'MAM']}}

    figures = [plt.subplots(1, 1, figsize=(fig_width, fig_height))
               for i in range(2)]

    for hidx, hem in enumerate(hems):
        fig, axis = figures[hidx]
        make_violinplot(metric, axis, hems, hem, fig_mult)

        fig.subplots_adjust(left=0.06, bottom=0.05,
                            right=0.98, top=0.98, hspace=0.0)
        fig.legend(bbox_to_anchor=(0.06, 0.05),
                   loc='lower left', borderaxespad=0.,
                   ncol=3)
        fig.savefig(f'plt_compare_dist_all_{hem}.{extn}')


def make_violinplot(metric, axis, hems, hem, fig_mult):
    """Make categorial plot for hemisphere on axis."""
    colors = ['#FF993F', '#ff7f0e',     # Orange
              '#5BA05B', '#0BA00B',     # Green
              '#6794B5', '#0069B5']     # Blue
    sns.violinplot(x='season', y='lat', hue='kind',
                   data=metric[metric.hem == hem], ax=axis,
                   inner='quart', dashpattern='-',
                   linewidth=0.75 * fig_mult, width=0.8,
                   palette=colors, order=hems[hem]['sea_order'],
                   bw=0.3, scale='width')

    # sns.boxplot(x='season', y='lat', hue='kind',
    #             data=metric[metric.hem == hem], ax=axis,
    #             linewidth=0.75 * fig_mult, width=0.8,
    #             palette=colors, order=hems[hem]['sea_order'])


    axis.set_yticks(hems[hem]['ticks'])
    axis.set_yticks(hems[hem]['min_ticks'], minor=True)
    axis.tick_params(pad=0.75)
    axis.set_xlabel('')
    axis.set_ylabel('Latitude [deg]')
    axis.grid(b=True, which='both', axis='y', ls='--', zorder=1)
    axis.set_ylim(hems[hem]['ylims'])
    axis.legend_.remove()
    axis.tick_params(axis='y', rotation=90)


def label(x, color, lab):
    """Put labels on gridplot axes."""
    axis = plt.gca()
    axis.text(0, .2, lab, fontweight="bold", color=color,
              ha="left", va="center", transform=axis.transAxes)


def gridplot(metric):
    """Make grid plot of multiple jetruns."""
    grd = sns.FacetGrid(metric[metric.hem == 'sh'],
                        row="season", hue="season")
    grd.map(sns.kdeplot, "lat", clip_on=False, lw=2, bw=1, kernel='triw')
    grd.map(plt.axhline, y=0, lw=1, clip_on=False, color='k')
    grd.map(label, "lat")
    grd.set_titles("")
    grd.set(yticks=[])
    grd.despine(bottom=True, left=True)
    plt.show()


if __name__ == "__main__":
    main(fig_mult=2.0)
