#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a summarising plot of a jet finding expedition."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


__author__ = 'Michael Kelleher'


def main(run_name=None, props=None):
    """Load jet run output, make a plot or two."""
    if props is None:
        props = {'pv': 2.0,
                 'fit': 6,
                 'y0': 10.0,
                 'yN': 65.0,
                 'zonal_reduce': 'mean',
                 'date_s': '1979-01-01',
                 'date_e': '2016-12-31'}

    if run_name is None:
        props['data'] = 'ERAI_MONTHLY_THETA_STJPV'
    else:
        props['data'] = run_name

    plt.rc('font', family='sans-serif', size=10)

    in_file = ('{data}_pv{pv:.1f}_fit{fit}_y0{y0:03.1f}_yN{yN:.1f}_'
               'z{zonal_reduce}_{date_s}_{date_e}'.format(**props))

    data = xr.open_dataset(f'{in_file}.nc')

    month_mean = data.groupby('time.month').mean()
    month_std = data.groupby('time.month').std()

    hem_info = {'nh': {'label': 'Northern Hemisphere', 'lat_r': (20, 50)},
                'sh': {'label': 'Southern Hemisphere', 'lat_r': (-50, -20)}}

    fig = plt.figure(figsize=(10, 7))
    axes = [plt.subplot2grid((2, 2), (0, 0)), plt.subplot2grid((2, 2), (0, 1)),
            plt.subplot2grid((2, 2), (1, 0), colspan=2)]

    for hidx, hem in enumerate(['sh', 'nh']):
        sct = axes[hidx].scatter(data[f'lat_{hem}'].values,
                                 data[f'theta_{hem}'].values,
                                 s=data[f'intens_{hem}'].values * 4,
                                 c=data[f'intens_{hem}'].values,
                                 cmap='inferno', vmin=0., vmax=45., alpha=0.4)

        # Hexbins? Maybe...
        # sct = axes[hidx].hexbin(data[f'lat_{hem}'].values,
        #                         data[f'theta_{hem}'].values,
        #                         gridsize=20, cmap='Blues', mincnt=1,
        #                         edgecolors='k', linewidths=0.1)

        axes[hidx].set_ylabel('Theta Position [K]')
        if hem == 'nh':
            cax = fig.add_axes([0.48, 0.51, 0.02, 0.4])
            cbar = plt.colorbar(sct, cax=cax, orientation='vertical')
            cbar.set_label('Jet intensity [m/s]')
            cax.yaxis.set_label_position('left')
            axes[hidx].tick_params(left='off', labelleft='off', labeltop='off',
                                   right='on', labelright='on')
            axes[hidx].ticklabel_format(axis='y', style='plain')
            axes[hidx].yaxis.set_label_position('right')

        axes[hidx].set_xlabel('Latitude Position [deg]')
        axes[hidx].set_title(hem_info[hem]['label'])
        axes[hidx].set_ylim([320, 370])
        axes[hidx].set_xlim(hem_info[hem]['lat_r'])

    ln_nh = axes[2].plot(month_mean.month, month_mean['lat_nh'].values,
                         'C0o-', label='NH')

    axes[2].fill_between(month_mean.month,
                         month_mean['lat_nh'] + month_std['lat_nh'],
                         month_mean['lat_nh'] - month_std['lat_nh'],
                         color='C0', alpha=0.6)

    axes[2].set_ylim(hem_info['nh']['lat_r'])
    axes[2].set_ylabel('NH Jet Latitude')

    axis_sh = axes[2].twinx()

    ln_sh = axis_sh.plot(month_mean.month, month_mean['lat_sh'], 'C1o-',
                         label='SH')
    axis_sh.fill_between(month_mean.month,
                         month_mean['lat_sh'] + month_std['lat_sh'],
                         month_mean['lat_sh'] - month_std['lat_sh'],
                         color='C1', alpha=0.6)
    axis_sh.set_ylim(hem_info['sh']['lat_r'])

    axes[2].tick_params('y', colors='C0')
    axis_sh.tick_params('y', colors='C1')

    axis_sh.set_xticks(np.arange(1, 13))
    axis_sh.set_xticklabels(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
    axis_sh.set_ylabel('SH Jet Latitude')

    lns = ln_nh + ln_sh
    labs = [l.get_label() for l in lns]
    axes[2].legend(lns, labs, loc=0)
    axes[2].grid(b=True, ls='--')
    axis_sh.invert_yaxis()

    data_name = ' '.join(props['data'].split('_'))
    plt.suptitle('{}: {pv} PVU, lat [{y0} - {yN}], '
                 'fit {fit}, zonal {zonal_reduce}'.format(data_name, **props))
    fig.subplots_adjust(left=0.06, bottom=0.04, right=0.93, top=0.91,
                        wspace=0.20, hspace=0.20)
    plt.savefig(f'{in_file}.png')


if __name__ == '__main__':
    for RNAME in ['NCEP_NCAR_MONTHLY_STJPV', 'NCEP_NCAR_DAILY_STJPV',
                  'ERAI_MONTHLY_THETA_STJPV', 'ERAI_DAILY_THETA_STJPV']:
        main(run_name=RNAME)
