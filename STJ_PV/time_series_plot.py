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

    for sea_idx in range(4):
        sub_axes = [axes[sea_idx]]
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
        lns = lines[0] + lines[1] + lines[2]
        labels = [l[0].get_label() for l in lines]
        sub_axes[0].legend(lns, labels)
        axes[sea_idx].set_title('{}'.format(sea_names[sea_idx]))

    plt.suptitle(hem.upper())
    plt.tight_layout()
    plt.savefig(f'plt_trend_{hem}.png')

def main():
    """Load a STJMetric output file, plot seasonal time series data."""
    in_f = 'ERAI_MONTHLY_THETA_STJPV_pv2.5_fit8_y010.0.nc'
    dset = xr.open_dataset(in_f)
    for hem in ['nh', 'sh']:
        plot_timeseries(dset, hem)

if __name__ == '__main__':
    main()
