#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot trends from Phil."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seaborn import despine

__author__ = 'Michael Kelleher, Penny Maher'
COLORS = {
    "MERRA2": "#0BA00B",
    "ERA-I": "#ff7f0e",
    "JRA55": "#0069B5",
    "CFSR": "#AF2525",
}

def invert_coords(coords, coord_key):
    """Invert coordinates to map labels to values."""
    inv_map = {val: key for key, val in coords[coord_key].items()}
    labels = [inv_map[key] for key in inv_map]
    keys = [key for key in inv_map]
    return labels, keys


def plot_data(info):
    """Load data, make error-bar plots of trends from Phil."""
    data = pd.read_csv(info['data_file'])

    fig_w = 17.4 / 2.54
    _, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * (9 / 16)))

    # Coordinates (axis, x of axis)
    coords = {
        'x': {'Monthly': 0, 'Daily': 1},
        'axis': {'Northern Hemisphere': 0, 'Southern Hemisphere': 1},
        'color': {'CFSR': 0, 'ERA-I': 1, 'JRA55': 2, 'MERRA2': 3},
        'xplus': {'CFSR': -0.2, 'ERA-I': -0.07, 'JRA55': 0.07, 'MERRA2': 0.2},
    }
    axkey = 'Hemisphere'
    xkey = 'Frequency'
    # leg_handles = {}
    for _, row in data.iterrows():
        ax_ix = coords['axis'][row[axkey]]
        x_ix = coords['x'][row[xkey]] + coords['xplus'][row['Reanalysis']]
        cix = coords['color'][row['Reanalysis']]

        sct_args = {
            'CFSR': {'marker': 'o', 's': info['ms'], 'zorder': 6},
            'JRA55': {'marker': 'o', 's': info['ms'], 'zorder': 7},
            'ERA-I': {'marker': 'o', 's': info['ms'], 'zorder': 8},
            'MERRA2': {'marker': 'o', 's': info['ms'], 'zorder': 9},
        }
        color = COLORS[row['Reanalysis']]
        if row['Upper'] * row['Lower'] > 0:
            sct_args[row['Reanalysis']]['c'] = color
        else:
            sct_args[row['Reanalysis']]['facecolor'] = 'white'
            sct_args[row['Reanalysis']]['edgecolor'] = color


        if ax_ix == 0 and coords['x'][row[xkey]] == 1:
            sct_args[row['Reanalysis']]['label'] = row['Reanalysis']

        axes[ax_ix].scatter(
            x_ix, row[info['data_name']], **sct_args[row['Reanalysis']]
        )

        axes[ax_ix].vlines(
            x_ix,
            row['Lower'],
            row['Upper'],
            color,
            lw=2.5,
            zorder=2,
        )
        capsize = 0.07

        axes[ax_ix].hlines(
            row['Lower'], x_ix - capsize, x_ix + capsize, color
        )
        axes[ax_ix].hlines(
            row['Upper'], x_ix - capsize, x_ix + capsize, color
        )

    xlabels, xticks = invert_coords(coords, 'x')
    axlabels, axidx = invert_coords(coords, 'axis')

    ax_key = ['(a)', '(b)']

    for idx, axis in enumerate(axes):
        axis.set_xlim([-0.5, 1.5])
        if info['data_name'] == 'Mean':
            axis.set_ylim(info['y_lim'][idx])
        else:
            axis.set_ylim(info['y_lim'])

        axis.grid(b=True, ls='--', lw=0.2)
        despine(ax=axis, left=False, bottom=True, offset=10)
        axis.axhline(0, color='k', lw=0.7)
        axis.xaxis.grid(b=False)
        axis.set_xticks(xticks)
        axis.set_xticklabels(xlabels)
        # axis.xaxis.set_ticks_position('none')
        axis.set_title('{} {}'.format(ax_key[idx], axlabels[idx]))

    axes[coords['axis']['Southern Hemisphere']].invert_yaxis()
    axes[0].set_ylabel(info['ylabel'])

    # _keys = [('Daily', 'Northern Hemisphere', reana) for reana in sct_args]
    axes[0].legend(
        bbox_to_anchor=info['legend'], frameon=False, framealpha=1.0
    )

    plt.tight_layout()
    plt.savefig('plt_{data_name}_all.pdf'.format(**info))


def main():
    """Setup info for plots, call plot creation function."""
    info = {
        'trends': {
            'data_name': 'Trend',
            'data_file': 'trends.csv',
            'ylabel': u'Latitude Trend [\u00b0 / decade]',
            'y_lim': [-0.97, 0.97],
            'legend': (0.99, 0.965),
            'ms': 65,
        },
        'means': {
            'data_name': 'Mean',
            'data_file': 'position.csv',
            'ylabel': u'Latitude Position [\u00b0]',
            'y_lim': ([28, 38], [-38, -28]),
            'legend': (0.4, 0.965),
            'ms': 30,
        },
    }

    plot_data(info['trends'])


if __name__ == '__main__':
    main()
