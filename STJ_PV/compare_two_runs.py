# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns

def boxplot(data):
    print(data)
    plt.boxplot(data.values)


#def main():
if __name__ == "__main__":
    """Compare jet latitudes of results from two different runs of stj_run."""

    file_info = {
        'NCEP-PV': {'file': './NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit12_y010.0.nc',
                    'label': 'NCEP PV'},
        'NCEP-Umax': {'file': './NCEP_NCAR_MONTHLY_HR_STJUMax_pres25000.0_y010.0.nc',
                      'label': 'NCEP U-max'},
        'ERAI-PV': {'file': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0.nc',
                    'label': 'ERAI Theta'},
        'ERAI-Umax': {'file': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc',
                      'label': 'ERAI Pres'}
    }
    # diffs = ['NCEP-PV', 'ERAI-PV']
    diffs = ['NCEP-PV', 'NCEP-Umax']

    d_in = {name: xr.open_dataset(file_info[name]['file'])
            for name in diffs}

    times = [d_in[ftype].time for ftype in diffs]
    dates = [pd.DatetimeIndex(nc.num2date(time.data[:], time.units)) for time in times]
    #dates = [time.values[:] for time in times]

    lat_nh = {in_f: d_in[in_f].variables['lat_nh'].data[:] for in_f in d_in}
    lat_sh = {in_f: d_in[in_f].variables['lat_sh'].data[:] for in_f in d_in}
    dates = [np.arange(0,len(lat_nh['ERAI-KP']),1), np.arange(0,len(lat_nh['ERAI-PV']),1)]
    min_shape = min([lat_nh[ft].shape[0] for ft in lat_nh])

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    for fix, in_f in enumerate(lat_nh):
        plt.plot(dates[fix], lat_nh[in_f], label=in_f)

    plt.title('NH')
    plt.legend()
    plt.grid(b=True, ls='--')

    plt.subplot(2, 2, 3)
    for fix, in_f in enumerate(lat_sh):
        plt.plot(dates[fix], lat_sh[in_f], label=in_f)
    plt.title('SH')
    plt.grid(b=True, ls='--')

    plt.subplot(2, 2, 2)
    plt.plot(dates[0][:min_shape],
             lat_nh[diffs[0]][:min_shape] - lat_nh[diffs[1]][:min_shape])
    plt.title('NH DIFF')
    plt.grid(b=True, ls='--')

    plt.subplot(2, 2, 4)
    plt.plot(dates[0][:min_shape],
             lat_sh[diffs[0]][:min_shape] - lat_sh[diffs[1]][:min_shape])
    plt.title('SH DIFF')
    plt.grid(b=True, ls='--')
    plt.tight_layout()

    plt.savefig('plt_compare_time_series_{}_{}.png'.format(*diffs))
    plt.close()

    nh_seas = {in_f: d_in[in_f]['lat_nh'].groupby('time.season') for in_f in diffs}
    sh_seas = {in_f: d_in[in_f]['lat_sh'].groupby('time.season') for in_f in diffs}


    diff_nh = nh_seas[diffs[0]].mean() - nh_seas[diffs[1]].mean()
    diff_sh = sh_seas[diffs[0]].mean() - sh_seas[diffs[1]].mean()
    bar_width = 0.35
    seasons = sh_seas[diffs[0]].mean().season.data.astype(str)
    index = np.arange(len(seasons))

    fig_width = 110 / 25.4
    fig_height = fig_width * (2 / (1 + np.sqrt(5)))
    font_size = 9
    plt.figure(figsize=(fig_width, fig_height))
    plt.bar(index, -diff_nh, bar_width, label='NH')
    plt.bar(index + bar_width, diff_sh, bar_width, label='SH')
    plt.xticks(index + bar_width / 2, seasons, fontsize=font_size)

    plt.ylabel(u'\u00b0 latitude', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.title('Equatorward bias of {} to {}'
              .format(file_info[diffs[0]]['label'], file_info[diffs[1]]['label']),
              fontsize=font_size)
    plt.subplots_adjust(left=0.19, bottom=0.12, right=0.97, top=0.89)
    plt.savefig('plt_compare_{}_{}.eps'.format(*diffs))
    plt.close()


    # Rearrange for plotting

    nh_lats = [d_in[in_f]['lat_nh'] for in_f in diffs]
    nh_diff = (nh_lats[0] - nh_lats[1])
    sh_lats = [d_in[in_f]['lat_sh'] for in_f in diffs]
    sh_diff = (sh_lats[0] - sh_lats[1])

    seasonal_nh = [{seas: lats[lats.groupby('time.season').groups[seas]].values
                    for seas in lats.groupby('time.season').groups} for lats in nh_lats]

    seasonal_nh = {diffs[0]: seasonal_nh[0], diffs[1]: seasonal_nh[1]}
    nh_data = {'lat': [], 'season': [], 'dtype': [], 'hem': []}
    for dtype in seasonal_nh:
        for season in seasonal_nh[dtype]:
            for lati in seasonal_nh[dtype][season]:
                nh_data['lat'].append(lati)
                nh_data['season'].append(season)
                nh_data['dtype'].append(dtype)
                nh_data['hem'].append('nh')

    seasonal_sh = [{seas: lats[lats.groupby('time.season').groups[seas]].values
                    for seas in lats.groupby('time.season').groups} for lats in sh_lats]
    seasonal_sh = {diffs[0]: seasonal_sh[0], diffs[1]: seasonal_sh[1]}
    sh_data = {'lat': [], 'season': [], 'dtype': [], 'hem': []}
    for dtype in seasonal_sh:
        for season in seasonal_sh[dtype]:
            for lati in seasonal_sh[dtype][season]:
                sh_data['lat'].append(lati)
                sh_data['season'].append(season)
                sh_data['dtype'].append(dtype)
                sh_data['hem'].append('sh')

    nh_data = pd.DataFrame(nh_data)
    sh_data = pd.DataFrame(sh_data)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_width * 2))
    sns.violinplot(x='season', y='lat', hue='dtype', data=nh_data, split=True,
                   inner='quart', palette={'NCEP-PV': 'C0', 'NCEP-Umax': 'C1'},
                   ax=axes[0], cut=0)
    sns.violinplot(x='season', y='lat', hue='dtype', data=sh_data, split=True,
                   inner='quart', palette={'NCEP-PV': 'C0', 'NCEP-Umax': 'C1'},
                   ax=axes[1], cut=0)
    for axis in axes:
        axis.legend_.remove()
    nh_tic = [30, 40, 50]
    axes[0].set_yticks(nh_tic)
    axes[0].set_yticklabels(['{:3.0f}'.format(i) for i in nh_tic])
    plt.savefig('plt_dist.png')
    plt.close()

    nh_seas_diff = {seas: nh_diff[nh_diff.groupby('time.season').groups[seas]]
                    for seas in nh_diff.groupby('time.season').groups}

    sh_seas_diff = {seas: sh_diff[sh_diff.groupby('time.season').groups[seas]]
                    for seas in sh_diff.groupby('time.season').groups}

    nh_seas_diff = pd.DataFrame(nh_seas_diff)
    sh_seas_diff = pd.DataFrame(sh_seas_diff)

    fig, axis = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    sns.violinplot(data=nh_seas_diff, ax=axis[0])
    axis[0].yaxis.grid(b=True, ls='--')
    sns.violinplot(data=sh_seas_diff, order=['JJA', 'SON', 'DJF', 'MAM'], ax=axis[1])
    axis[1].yaxis.grid(b=True, ls='--')

    for axi in axis:
        axi.set_xticks([0, 1, 2, 3])
        axi.set_xticklabels(['Winter', 'Spring', 'Summer', 'Autumn'], fontsize=font_size)
        #axi.set_yticks([-5, 0, 25])
        #axi.set_ylim([-10, 35])

    axis[0].set_ylabel(u'\u00b0 latitude [NH]', fontsize=font_size)
    axis[1].set_ylabel(u'\u00b0 latitude [SH]', fontsize=font_size)
    fig.suptitle('Latitude difference of {} to {}'
                 .format(file_info[diffs[0]]['label'], file_info[diffs[1]]['label']),
                 fontsize=font_size)
    fig.subplots_adjust(left=0.16, bottom=0.12, right=0.97, top=0.89, hspace=0.0)
    #plt.show()
    plt.savefig('plt_compare_boxplot_{}_{}.png'.format(*diffs))


#if __name__ == "__main__":
#    main()
