# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
#plt.style.use('ggplot')
import pdb 

def main():
    """Compare jet latitudes of results from two different runs of stj_run."""
    #files_in = {'minlat': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0_minlat.nc',
    #            'ushear': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0.nc'}
    #files_in = {'Theta': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0.nc',
    #            'Press': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc'}
    #files_in = {'minlat': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc',
    #            'strflat': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0_psimax.nc'}
    #files_in = {'cb4': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit4_y010.0.nc',
    #            'cb8': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0.nc'}
    #files_in = {'ERA': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc',
    #            'NCEP-HR': './NCEP_NCAR_MONTHLY_HR_STJPV_pv2.0_fit12_y010.0.nc'}
    #files_in = {'ERAI': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0.nc',
    #            'NCEP': './NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit12_y010.0.nc'}
    #files_in = {'ERAI-PV': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc',
    #            'ERAI-Umax': './ERAI_PRES_STJUMax_pres25000.0_y010.0_1979-01-01_2016-12-31.nc'}
    files_in = {'ERAI-KP': './ERAI_PRES_KangPolvani_1979-01-01_2015-12-31.nc',
                'ERAI-PV': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0_1979-01-01_2015-12-31.nc'}

    ftypes = sorted(files_in.keys())

    d_in = {in_f: xr.open_dataset(files_in[in_f], decode_times=False)
            for in_f in files_in}


    #times = [d_in[ftype].time for ftype in ftypes]
    #dates = [pd.DatetimeIndex(nc.num2date(time.data[:], time.units)) for time in times]
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
             lat_nh[ftypes[0]][:min_shape] - lat_nh[ftypes[1]][:min_shape])
    plt.title('NH DIFF')
    plt.grid(b=True, ls='--')


    plt.subplot(2, 2, 4)
    plt.plot(dates[0][:min_shape],
             lat_sh[ftypes[0]][:min_shape] - lat_sh[ftypes[1]][:min_shape])
    plt.title('SH DIFF')
    plt.grid(b=True, ls='--')
    plt.tight_layout()
    plt.savefig('plt_compare_time_series_{}_{}.png'.format(*files_in.keys()))
    plt.show()
    plt.close()
    pdb.set_trace()
    #diffs = ['NCEP-PV', 'NCEP-Umax']
    #labels = {'NCEP-PV': 'PV', 'NCEP-Umax': 'u max'}
    diffs = ['ERAI-PV', 'ERAI-Umax']
    labels = {'ERAI-PV': 'PV', 'ERAI-Umax': 'u max'}
    #diffs = ['NCEP', 'ERAI']
    #labels = {'NCEP': 'NCEP', 'ERAI': 'ERA-int'}
    d_in = {in_f: xr.open_dataset(files_in[in_f]) for in_f in files_in}

    nh_seas = {in_f: d_in[in_f]['lat_nh'].groupby('time.season') for in_f in files_in}
    sh_seas = {in_f: d_in[in_f]['lat_sh'].groupby('time.season') for in_f in files_in}

    diff_nh = nh_seas[diffs[0]].mean() - nh_seas[diffs[1]].mean()
    diff_sh = sh_seas[diffs[0]].mean() - sh_seas[diffs[1]].mean()
    bar_width = 0.35
    seasons = sh_seas[diffs[0]].mean().season.data.astype(str)
    index = np.arange(len(seasons))

    fig_width = 84 / 25.4
    fig_height = fig_width * (2 / (1 + np.sqrt(5)))
    font_size = 9
    plt.figure(figsize=(fig_width, fig_height))
    plt.bar(index, -diff_nh, bar_width, label='NH')
    plt.bar(index + bar_width, diff_sh, bar_width, label='SH')
    plt.xticks(index + bar_width/2, seasons, fontsize=font_size)

    plt.ylabel(u'\u00b0 latitude', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.title('Equatorward bias of {} to {}'
              .format(labels[diffs[0]], labels[diffs[1]]), fontsize=font_size)
    plt.subplots_adjust(left=0.19, bottom=0.12, right=0.97, top=0.89)
    plt.savefig('plt_compare_{}_{}.eps'.format(*files_in.keys()))


if __name__ == "__main__":
    main()
