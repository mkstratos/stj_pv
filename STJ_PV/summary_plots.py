#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a summarising plot of a jet finding expedition."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml

plt.style.use('fivethirtynine')

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

    plt.rc('font', family='sans-serif', size=9)

    in_file = ('{data}_pv{pv:.1f}_fit{fit}_y0{y0:03.1f}_yN{yN:.1f}_'
               'z{zonal_reduce}_{date_s}_{date_e}'.format(**props))

    data = xr.open_dataset(f'./jet_out/{in_file}.nc')

    month_mean = data.groupby('time.month').mean()
    month_std = data.groupby('time.month').std()

    fig_width = 17.4 / 2.54
    figh_height = fig_width * 0.6
    fig = plt.figure(figsize=(fig_width, figh_height))
    axes = [plt.subplot2grid((2, 2), (0, 0)), plt.subplot2grid((2, 2), (0, 1)),
            plt.subplot2grid((2, 2), (1, 0), colspan=2)]

    for hidx, hem in enumerate(['sh', 'nh']):
        sct = axes[hidx].scatter(data[f'lat_{hem}'].values,
                                 data[f'theta_{hem}'].values,
                                 s=0.3 * data[f'intens_{hem}'].values**2,
                                 c=data[f'intens_{hem}'].values, marker='.',
                                 cmap='inferno', vmin=0., vmax=45., alpha=0.75)

        # Hexbins? Maybe...
        # sct = axes[hidx].hexbin(data[f'lat_{hem}'].values,
        #                         data[f'theta_{hem}'].values,
        #                         gridsize=20, cmap='Blues', mincnt=1,
        #                         edgecolors='k', linewidths=0.1)

        axes[hidx].set_ylabel('Theta Position [K]')
        if hem == 'nh':
            cax = fig.add_axes([0.49, 0.51, 0.02, 0.4])
            cbar = plt.colorbar(sct, cax=cax, orientation='vertical')
            cbar.set_label('Jet intensity [m/s]')
            cax.yaxis.set_label_position('left')
            axes[hidx].tick_params(left='off', labelleft='off', labeltop='off',
                                   right='on', labelright='on')
            axes[hidx].ticklabel_format(axis='y', style='plain')
            axes[hidx].yaxis.set_label_position('right')

        axes[hidx].set_xlabel('Latitude Position [deg]')
        axes[hidx].set_title(HEM_INFO[hem]['label'])
        axes[hidx].set_ylim([320, 375])
        axes[hidx].set_xlim(HEM_INFO[hem]['lat_r'])

    ln_nh = axes[2].plot(month_mean.month, month_mean['lat_nh'].values,
                         'C0o-', label='NH', zorder=5)
    axes[2].fill_between(month_mean.month,
                         month_mean['lat_nh'] + month_std['lat_nh'],
                         month_mean['lat_nh'] - month_std['lat_nh'],
                         color='C0', alpha=0.4, zorder=4)
    axes[2].fill_between(month_mean.month,
                         month_mean['lat_nh'] + 2 * month_std['lat_nh'],
                         month_mean['lat_nh'] - 2 * month_std['lat_nh'],
                         color='C0', alpha=0.1, zorder=3)

    axes[2].set_ylim(HEM_INFO['nh']['lat_r'])
    axes[2].set_ylabel('NH Jet Latitude')

    axis_sh = axes[2].twinx()

    ln_sh = axis_sh.plot(month_mean.month, month_mean['lat_sh'], 'C1o-',
                         label='SH')

    axis_sh.fill_between(month_mean.month,
                         month_mean['lat_sh'] + month_std['lat_sh'],
                         month_mean['lat_sh'] - month_std['lat_sh'],
                         color='C1', alpha=0.4)

    axis_sh.fill_between(month_mean.month,
                         month_mean['lat_sh'] + 2 * month_std['lat_sh'],
                         month_mean['lat_sh'] - 2 * month_std['lat_sh'],
                         color='C1', alpha=0.1)
    axis_sh.set_ylim(HEM_INFO['sh']['lat_r'])

    axes[2].tick_params('y', colors='C0')
    axis_sh.tick_params('y', colors='C1')

    axis_sh.set_xticks(np.arange(1, 13))
    axis_sh.set_xticklabels(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
    axis_sh.set_ylabel('SH Jet Latitude')

    lns = ln_nh + ln_sh
    labs = [l.get_label() for l in lns]
    axes[2].legend(lns, labs, loc=0, frameon=False)
    axes[2].set_axisbelow(True)
    axes[2].grid(b=True, zorder=-1)
    axes[2].xaxis.grid(b=False)
    axis_sh.grid(b=False)
    axis_sh.invert_yaxis()

    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.92, top=0.94,
                        wspace=0.23, hspace=0.28)
    out_file = in_file.replace('.', 'p')
    plt.savefig(f'{out_file}.{EXTN}')
    pair_grid(data, out_file)
    summary_table(data)


def pair_grid(data, out_file):
    """Make PairGrid plot for each hemisphere."""
    dframe = data.to_dataframe()
    nh_vars = [f'{var}_nh' for var in ['theta', 'lat', 'intens']]
    sh_vars = [f'{var}_sh' for var in ['theta', 'lat', 'intens']]
    df_nh = dframe[nh_vars]
    df_sh = dframe[sh_vars]

    for hem, dfi in [('nh', df_nh), ('sh', df_sh)]:
        dfi = dfi.rename(index=str,
                         columns={f'lat_{hem}': 'Latitude [deg]',
                                  f'theta_{hem}': 'Theta [K]',
                                  f'intens_{hem}': 'Intensity [m/s]'})
        dfi['season'] = SEAS[pd.DatetimeIndex(dfi.index).month].astype(str)
        grd = sns.PairGrid(dfi, diag_sharey=False, hue='season',
                           palette=HEM_INFO[hem]['pal'])
        grd.fig.set_size_inches(17.4 / 2.54, 17.4 / 2.54)
        grd.map_lower(plt.scatter, marker='o', alpha=0.8, s=6.0)
        grd.map_diag(sns.kdeplot, lw=2)
        plt.suptitle(f'{HEM_INFO[hem]["label"]} Properties')
        grd.fig.subplots_adjust(top=0.92)
        grd.add_legend(frameon=False)

        # Because we've separated by hue, have to do KDE plots separately
        for i, j in zip(*np.triu_indices_from(grd.axes, 1)):
            sns.kdeplot(dfi[grd.x_vars[j]], dfi[grd.y_vars[i]],
                        shade=True, cmap='Reds', legend=False,
                        shade_lowest=False, ax=grd.axes[i, j])
        plt.savefig(f'plt_grid_{hem}_{out_file}.{EXTN}')
        plt.clf()
        plt.close()


def summary_table(data_in):
    """Compute monthly, seasonal, and annual mean for NH and SH."""
    data_seasonal = data_in.groupby('time.season').mean()
    data_monthly = data_in.groupby('time.month').mean()
    out_file = yaml.load(data_in.run_props)['output_file']
    out_file = out_file.replace('.', 'p')

    with open(f'seasonal_stats_{out_file}.tex', 'w') as fout:
        fout.write(data_seasonal.to_dataframe().to_latex())

    with open(f'monthly_stats_{out_file}.tex', 'w') as fout:
        fout.write(data_monthly.to_dataframe().to_latex())

    with open(f'annual_stats_{out_file}.tex', 'w') as fout:
        fout.write(data_in.to_dataframe().mean().to_latex())


# Color map for seasons
EXTN = 'pdf'

COLS = {'summer': sns.xkcd_rgb['red'],
        'winter': sns.xkcd_rgb['denim blue'],
        'spring': sns.xkcd_rgb['medium green'],
        'autumn': sns.xkcd_rgb['pumpkin orange']}

HEM_INFO = {'nh': {'label': 'Northern Hemisphere', 'lat_r': (19, 51),
                   'pal': [COLS['winter'], COLS['spring'],
                           COLS['summer'], COLS['autumn']]},
            'sh': {'label': 'Southern Hemisphere', 'lat_r': (-51, -19),
                   'pal': [COLS['summer'], COLS['autumn'],
                           COLS['winter'], COLS['spring']]}}

SEAS = np.array([None, 'DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA',
                 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF'])

if __name__ == '__main__':
    DATASETS = ['NCEP_NCAR_MONTHLY_STJPV', 'NCEP_NCAR_DAILY_STJPV',
                'ERAI_MONTHLY_THETA_STJPV', 'ERAI_DAILY_THETA_STJPV']

    for RNAME in DATASETS[2:]:
        main(run_name=RNAME)
