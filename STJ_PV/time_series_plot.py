#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create seasonally separated time series plots of STJMetric output.
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

__author__ = 'Michael Kelleher'


def plot_timeseries(dset, hem='nh'):
    """Plot seasonal timeseries in all three outputs of a JetMetric."""
    in_vars = ['theta_{}'.format(hem), 'intens_{}'.format(hem), 'lat_{}'.format(hem)]
    ranges = {'theta': [370, 330], 'intens': [15, 40], 'lat': [25, 45]}

    _, axes = plt.subplots(2, 2, figsize=(19, 7))
    axes = axes.ravel()

    if hem == 'nh':
        seas_range = [0, 2, 1, 3]
    else:
        seas_range = [1, 3, 0, 2]

    for axx, sea_idx in enumerate(seas_range):
        sub_axes = [axes[axx]]
        for _ in range(2):
            sub_axes.append(sub_axes[0].twinx())

        lines = []
        for idx, var in enumerate(in_vars):
            seasons = [i for i in iter(dset[var].groupby('time.season'))]
            sea_names = [sea[0] for sea  in seasons]

            means = [np.abs((seas[1].values[::3] + seas[1].values[1::3] +
                             seas[1].values[2::3])) / 3 for seas in seasons]

            time_index = pd.DatetimeIndex(seasons[sea_idx][1].time.values)
            times = np.arange(time_index[0].year, time_index[-1].year + 1)
            lines.append(sub_axes[idx].plot(times, means[sea_idx],
                                            'C{}o-'.format(idx), label=var))

            sub_axes[idx].set_xticks(times[::2])
            sub_axes[idx].yaxis.set_tick_params(labelcolor=f'C{idx}', color=f'C{idx}')
            sub_axes[idx].set_ylim(ranges[var.split('_')[0]])

        sub_axes[-1].spines['right'].set_position(('outward', 25))
        if axx == 1:
            lns = lines[0] + lines[1] + lines[2]
            labels = [l[0].get_label() for l in lines]
            sub_axes[0].legend(lns, labels, ncol=3)
        axes[axx].set_title('{}'.format(sea_names[sea_idx]))

    plt.suptitle(hem.upper())
    plt.tight_layout()
    plt.savefig(f'plt_trend_{hem}.png')

def main():
    """Load a STJMetric output file, plot seasonal time series data."""
    in_f = 'ERAI_PRES_STJPV_pv2.0_fit10_y010.0_1979-01-01_2016-12-31.nc'
    dset = xr.open_dataset(in_f)
    for hem in ['nh', 'sh']:
        plot_timeseries(dset, hem)

if __name__ == '__main__':
    main()
