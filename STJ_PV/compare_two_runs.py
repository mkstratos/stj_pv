# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def main():
    """Compare jet latitudes of results from two different runs of stj_run."""
    #files_in = {'minlat': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0_minlat.nc',
    #            'ushear': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0.nc'}
    files_in = {'Theta': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0.nc',
                'Press': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc'}
    #files_in = {'minlat': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc',
    #            'strflat': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0_psimax.nc'}
    #files_in = {'cb4': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit4_y010.0.nc',
    #            'cb8': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0.nc'}
    ftypes = sorted(files_in.keys())

    d_in = {in_f: nc.Dataset(files_in[in_f], 'r') for in_f in files_in}

    time = d_in[ftypes[0]].variables['time']
    date = pd.DatetimeIndex(nc.num2date(time[:], time.units))
    lat_nh = {in_f: d_in[in_f].variables['lat_nh'][:] for in_f in d_in}
    lat_sh = {in_f: d_in[in_f].variables['lat_sh'][:] for in_f in d_in}

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    for in_f in lat_nh:
        plt.plot(date, lat_nh[in_f], label=in_f)
    plt.title('NH')
    plt.legend()
    plt.grid(b=True)

    plt.subplot(2, 2, 2)
    plt.plot(date, lat_nh[ftypes[0]] - lat_nh[ftypes[1]])
    plt.title('NH DIFF')
    plt.grid(b=True)

    plt.subplot(2, 2, 3)
    for in_f in lat_sh:
        plt.plot(date, lat_sh[in_f], label=in_f)
    plt.title('SH')
    plt.grid(b=True)

    plt.subplot(2, 2, 4)
    plt.plot(date, lat_sh[ftypes[0]] - lat_sh[ftypes[1]])
    plt.title('SH DIFF')
    plt.grid(b=True)
    plt.tight_layout()
    plt.savefig('plt_compare_{}_{}.png'.format(*files_in.keys()))


if __name__ == "__main__":
    main()
