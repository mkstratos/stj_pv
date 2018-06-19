#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot trends from Phil."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

__author__ = 'Michael Kelleher, Penny Maher'


def main():
    """Load data, make error-bar plots of trends from Phil."""
    data = pd.read_csv('trends.csv')

    fig_w = 17.4 / 2.54
    _, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * (9 / 16)))

    # Coordinates (axis, x of axis)
    coords = {'x': {'Monthly': 0, 'Daily': 1},
              'axis': {'NH': 0, 'SH': 1},
              'color': {'NCEP': 2, 'ERAI': 6}}

    for _, row in data.iterrows():
        ax_ix = coords['axis'][row['Hemisphere']]
        x_ix = coords['x'][row['Frequency']]
        cix = coords['color'][row['Reanalysis']]

        sct_args = {'marker': 'o', 's': 80, 'zorder': 6, 'c': f'C{cix}'}
        if row['Upper'] * row['Lower'] > 0:
            sct_args['edgecolor'] = 'k'
        else:
            sct_args['edgecolor'] = 'face'

        if ax_ix == 1 and x_ix == 0:
            sct_args['label'] = row['Reanalysis']

        axes[ax_ix].scatter(x_ix, row['Trend'], **sct_args)
        axes[ax_ix].vlines(x_ix, row['Lower'], row['Upper'],
                           f'C{cix}', lw=2.5, zorder=2)
        capsize = 0.07

        axes[ax_ix].hlines(row['Lower'], x_ix - capsize, x_ix + capsize,
                           f'C{cix}')
        axes[ax_ix].hlines(row['Upper'], x_ix - capsize, x_ix + capsize,
                           f'C{cix}')

    for axis in axes:
        axis.set_xlim([-0.5, 1.5])
        y_max = np.max(np.abs(axis.get_ylim()))
        axis.set_ylim([-y_max, y_max])
        axis.grid(b=True, ls='--', lw=0.5)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(['Monthly', 'Daily'])

    axes[0].set_title('Northern Hemisphere')
    axes[1].set_title('Southern Hemisphere')
    axes[0].set_ylabel(u'Latitude Trend [\u00b0 / decade]')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('plt_trends_all.pdf')


if __name__ == '__main__':
    main()
