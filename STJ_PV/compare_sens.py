# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
plt.style.use('ggplot')


def main(param_name, param_vals):
    """
    Compare jet latitudes of results from sensitivity runs of stj_run.

    Parameters
    ----------
    param_name : string
        Name of sensitivity parameter
    param_vals : iterable
        Array or list of values that `param_name` takes

    """
    opts = {'run_type': 'ERAI_MONTHLY_THETA_STJPV', 'fit': 12, 'y0': 10, 'pv_lev': 2.0}
    opts.pop(param_name)
    opts_var = [{param_name: p_val} for p_val in param_vals]
    file_fmt = '{run_type}_pv{pv_lev:.1f}_fit{fit:.0f}_y0{y0:03.1f}.nc'
    files_in  = [file_fmt.format(**var, **opts) for var in opts_var]
    print(files_in)
    d_in = xr.concat([xr.open_dataset(in_f) for in_f in files_in], dim=param_name)
    d_in[param_name] = param_vals

    time = d_in.time

    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    axes = axes.ravel()

    for idx, param in enumerate(d_in[param_name]):
        axes[0].plot(time, d_in.lat_nh[idx, :], label='{:.1f}'.format(float(param)))
        axes[1].plot(time, d_in.lat_sh[idx, :], label='{:.1f}'.format(float(param)))

    axes[0].set_title('NH')
    axes[0].legend()
    axes[0].grid(b=True)

    axes[1].set_title('SH')
    axes[1].legend()
    axes[1].grid(b=True)

    plt.tight_layout()
    plt.savefig('plt_compare_{}.png'.format(param_name))
    plt.close()

    nh_sm = d_in.lat_nh.groupby('time.season').mean(axis=1)
    sh_sm = d_in.lat_sh.groupby('time.season').mean(axis=1)

    fig, axis = plt.subplots(1, 2, figsize=(12, 5))
    for snx, season in enumerate(nh_sm.season):
        axis[0].plot(param_vals, nh_sm[:, snx], 'o-', label=str(season.data))
        axis[1].plot(param_vals, sh_sm[:, snx], 'o-', label=str(season.data))

    axis[0].set_xlabel('PV Level [PVU]')
    axis[0].set_ylabel('Mean Jet Latitude')
    axis[0].legend()
    axis[0].set_title('Northern Hemisphere')
    axis[1].set_xlabel('PV Level [PVU]')
    axis[1].legend()
    axis[1].set_title('Southern Hemisphere')
    plt.suptitle('Seasonal Mean Jet Latitude')
    plt.tight_layout()
    plt.savefig('plt_season_mean_{}.png'.format(param_name))


if __name__ == "__main__":
    main('pv_lev', np.arange(1.0, 5.5, 0.5))
    main('fit', np.arange(4, 12))
