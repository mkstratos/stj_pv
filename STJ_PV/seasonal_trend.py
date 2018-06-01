#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as sts
import netCDF4 as nc
import pandas as pd

__author__ = 'Michael Kelleher'


def main(time_freq):
    """

    """
    plt.rc('font', size=9)
    files = {'monthly': ('./jet_out/ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0_'
                         '1979-01-01_2016-12-31.nc'),
             'daily': ('./jet_out/ERAI_DAILY_THETA_STJPV_pv2.0_fit8_y010.0_'
                       '1979-01-01_2016-12-31.nc'),
             'ncep': ('./jet_out/NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit8_y010.0_'
                      '1979-01-01_2016-12-31.nc'),
             'merra': ('./jet_out/MERRA_MONTHLY_STJPV_pv2.0_fit8_y010.0_'
                       '1979-01-01_2015-12-31.nc')
             }

    in_file = files[time_freq]
    data_in = xr.open_dataset(in_file)

    cols = {'nh': {'DJF': 0, 'MAM': 3, 'JJA': 2, 'SON': 1},
            'sh': {'DJF': 2, 'MAM': 1, 'JJA': 0, 'SON': 3}}
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    seasonal = data_in.resample(freq='Q-MAY', dim='time')

    hem_desc = {'nh': 'Northern hemisphere', 'sh': 'Southern hemisphere'}
    jet_vars = {'lat': 'jet latitude [deg]', 'intens': 'jet intensity [m/s]',
                'theta': 'theta level [K]'}

    ylims = {'nh': (24, 48), 'sh': (-48, -24)}
    yticks = {'nh': np.arange(20, 46, 4), 'sh': -np.arange(20, 46, 4)[::-1]}
    confidence = 0.95

    for var in jet_vars:
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        for hidx, hem in enumerate(hem_desc):
            for sidx, season in enumerate(seasons):
                hvar = f'{var}_{hem}'

                years = pd.DatetimeIndex(seasonal.time.values[sidx:-1:4]).year
                trend = sts.linregress(years, seasonal[hvar].values[sidx:-1:4])

                if trend.pvalue < 1 - confidence:
                    axes[hidx].plot(years, trend.slope * years + trend.intercept,
                                    f'C{cols[hem][season]}--', lw=0.8)
                    label = (f'{season} {trend.slope:.3f}'
                             f' ({trend.rvalue**2:.3f}, {trend.pvalue:.2f})')
                else:
                    label = (f'{season}')
                if var == 'lat':
                    axes[hidx].set_yticks(yticks[hem])
                    axes[hidx].set_ylim(ylims[hem])
                axes[hidx].plot(years, seasonal[hvar].values[sidx:-1:4],
                                f'C{cols[hem][season]}', label=label, lw=1.5)
                axes[hidx].set_ylabel(f'{hem} {jet_vars[var]} trend')
                axes[hidx].tick_params(axis='y', rotation=90)
                axes[hidx].grid(b=True, ls='--', lw=0.6)
                axes[hidx].legend()

            axes[1].set_xlabel('Season end year')
            fig.suptitle(f'Trend of {jet_vars[var]}')
            plt.tight_layout()
            fig.savefig(f'plt_{var}_trend_{time_freq}.png')
            fig.subplots_adjust(top=0.93)
            plt.close()


if __name__ == '__main__':
    main('merra')
