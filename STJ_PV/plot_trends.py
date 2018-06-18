#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot trends from Phil."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# plt.style.use('fivethirtyeight')

__author__ = 'Michael Kelleher'


def main():
    """
    Load dataframe of trends, make plots.
    """
    data = pd.read_csv('trends.csv')
    fig_w = 17.4 / 2.54
    fig_h = fig_w * (9 / 16)
    hem_name = {'NH': 'Northern Hemisphere', 'SH': 'Southern Hemisphere'}
    _, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for hix, hem in enumerate(['NH', 'SH']):
        df_h = data[data['Hemisphere'] == hem]
        lower = df_h.Lower
        upper = df_h.Upper
        trend = df_h.Trend
        x_idx = np.arange(trend.shape[0])
        labels = ['{Reanalysis} {Freqency}'.format(**df_h.loc[i]) for i in df_h.index]
        for xix, df_i in enumerate(df_h.index):
            # axes[hix].plot(xix, trend[df_i], 'o', ms=6.0)
            sct_args = {'marker': 'o', 's': 80, 'zorder': 6}

            if upper[df_i] * lower[df_i] > 0:
                sct_args['edgecolor'] = 'k'
            else:
                sct_args['edgecolor'] = 'face'

            axes[hix].scatter(xix, trend[df_i], **sct_args)
            axes[hix].vlines(xix, lower[df_i], upper[df_i], 'k', lw=2.5, zorder=2)

        axes[hix].set_xticks(x_idx)
        axes[hix].set_xticklabels(labels, rotation=30)
        axes[hix].set_title(f'{hem_name[hem]}')
        axes[hix].grid(b=True, ls='-.', lw=0.5)
        y_max = np.max(np.abs(axes[hix].get_ylim()))
        axes[hix].set_ylim([-y_max, y_max])

    axes[0].set_ylabel('Latitude trend [deg / decade]')
    plt.tight_layout()
    plt.savefig('plt_trends_all.pdf')


if __name__ == '__main__':
    main()
