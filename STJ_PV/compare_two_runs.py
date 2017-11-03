# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
#plt.style.use('ggplot')


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
    files_in = {'NCEP-PV': './NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit12_y010.0.nc',
                'NCEP-Umax': './NCEP_NCAR_MONTHLY_HR_STJUMax_pres25000.0_y010.0.nc'}

    ftypes = sorted(files_in.keys())

    d_in = {in_f: xr.open_dataset(files_in[in_f], decode_times=False)
            for in_f in files_in}

    times = [d_in[ftype].time for ftype in ftypes]
    dates = [pd.DatetimeIndex(nc.num2date(time.data[:], time.units)) for time in times]
    lat_nh = {in_f: d_in[in_f].variables['lat_nh'].data[:] for in_f in d_in}
    lat_sh = {in_f: d_in[in_f].variables['lat_sh'].data[:] for in_f in d_in}

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
    plt.savefig('plt_compare_{}_{}.png'.format(*files_in.keys()))
    plt.show()

if __name__ == "__main__":
    main()
