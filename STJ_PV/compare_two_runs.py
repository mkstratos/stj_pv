# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def main():
    """Compare jet latitudes of results from two different runs of stj_run."""
    files_in = {'minlat': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0_minlat.nc',
                'ushear': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit12_y010.0.nc'}
    d_in = {in_f: nc.Dataset(files_in[in_f], 'r') for in_f in files_in}

    time = d_in['minlat'].variables['time'][:]
    lat_nh = {in_f: d_in[in_f].variables['theta_nh'][:] for in_f in d_in}
    lat_sh = {in_f: d_in[in_f].variables['theta_sh'][:] for in_f in d_in}

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    for in_f in lat_nh:
        plt.plot(time, lat_nh[in_f], label=in_f)
    plt.title('NH')
    plt.legend()
    plt.grid(b=True)

    plt.subplot(2, 2, 2)
    plt.plot(time, lat_nh['minlat'] - lat_nh['ushear'])
    plt.title('NH DIFF')
    plt.grid(b=True)

    plt.subplot(2, 2, 3)
    for in_f in lat_sh:
        plt.plot(time, lat_sh[in_f], label=in_f)
    plt.title('SH')
    plt.grid(b=True)

    plt.subplot(2, 2, 4)
    plt.plot(time, lat_sh['minlat'] - lat_sh['ushear'])
    plt.title('SH DIFF')
    plt.grid(b=True)
    plt.tight_layout()
    plt.savefig('plt_compare_{}_{}.png'.format(*files_in.keys()))


if __name__ == "__main__":
    main()
