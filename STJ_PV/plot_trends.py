#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot trends from Phil."""
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import despine
__author__ = 'Michael Kelleher, Penny Maher'


def invert_coords(coords, coord_key):
    """Invert coordinates to map labels to values."""
    inv_map = {val: key for key, val in coords[coord_key].items()}
    labels = [inv_map[key] for key in inv_map]
    keys = [key for key in inv_map]
    return labels, keys


def main():
    """Load data, make error-bar plots of trends from Phil."""
    data = pd.read_csv('trends.csv')

    fig_w = 17.4 / 2.54
    _, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * (9 / 16)))

    # Coordinates (axis, x of axis)
    coords = {'x': {'Monthly': 0, 'Daily': 1},
              'axis': {'Northern Hemisphere': 0, 'Southern Hemisphere': 1},
              'color': {'NCEP': 0, 'ERAI': 1}}
    axkey = 'Hemisphere'
    xkey = 'Frequency'

    for _, row in data.iterrows():
        ax_ix = coords['axis'][row[axkey]]
        x_ix = coords['x'][row[xkey]]
        cix = coords['color'][row['Reanalysis']]

        sct_args = {'marker': 'o', 's': 70, 'zorder': 6}
        if row['Upper'] * row['Lower'] > 0:
            sct_args['c'] = f'C{cix}'
        else:
            sct_args['facecolor'] = 'white'
            sct_args['edgecolor'] = f'C{cix}'

        if ax_ix == 1 and x_ix == 0:
            sct_args['label'] = row['Reanalysis']

        axes[ax_ix].scatter(x_ix, row['Trend'], **sct_args)
        axes[ax_ix].vlines(x_ix, row['Lower'], row['Upper'],
                           f'C{cix}', lw=2.5, zorder=2)
        capsize = 0.1

        axes[ax_ix].hlines(row['Lower'], x_ix - capsize, x_ix + capsize,
                           f'C{cix}')
        axes[ax_ix].hlines(row['Upper'], x_ix - capsize, x_ix + capsize,
                           f'C{cix}')

    xlabels, xticks = invert_coords(coords, 'x')
    axlabels, axidx = invert_coords(coords, 'axis')

    ax_key = ['(a)', '(b)']

    for idx, axis in enumerate(axes):
        axis.set_xlim([-0.5, 1.5])
        y_max = 0.67    # np.max(np.abs(axis.get_ylim()))
        axis.set_ylim([-y_max, y_max])
        axis.grid(b=True, ls='--', lw=0.2)
        despine(ax=axis, left=False, bottom=True, offset=10)
        axis.axhline(0, color='k', lw=0.7)
        axis.xaxis.grid(b=False)
        axis.set_xticks(xticks)
        axis.set_xticklabels(xlabels)
        axis.xaxis.set_ticks_position('none')
        axis.set_title(f'{ax_key[idx]} {axlabels[idx]}')

    axes[coords['axis']['Southern Hemisphere']].invert_yaxis()
    axes[0].set_ylabel(u'Latitude Trend [\u00b0 / decade]')
    axes[1].legend(bbox_to_anchor=(.99, .965), frameon=False)

    plt.tight_layout()
    plt.savefig('plt_trends_all.pdf')


if __name__ == '__main__':
    main()
