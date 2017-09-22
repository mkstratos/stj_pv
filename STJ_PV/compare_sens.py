# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#plt.style.use('ggplot')


def main(param_name, param_vals, var_name='lat'):
    """
    Compare jet latitudes of results from sensitivity runs of stj_run.

    Parameters
    ----------
    param_name : string
        Name of sensitivity parameter
    param_vals : iterable
        Array or list of values that `param_name` takes

    """
    opts = {'run_type': 'ERAI_MONTHLY_THETA_STJPV', 'fit': 8, 'y0': 10, 'pv_lev': 2.0}
    opts.pop(param_name)
    opts_var = [{param_name: p_val} for p_val in param_vals]
    file_fmt = '{run_type}_pv{pv_lev:.1f}_fit{fit:.0f}_y0{y0:03.1f}.nc'
    files_in  = [file_fmt.format(**var, **opts) for var in opts_var]
    d_in = xr.concat([xr.open_dataset(in_f) for in_f in files_in], dim=param_name)
    d_in[param_name] = param_vals

    time = d_in.time

    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    axes = axes.ravel()

    for idx, param in enumerate(d_in[param_name]):
        axes[0].plot(time, d_in['{}_nh'.format(var_name)][idx, :],
                     label='{:.1f}'.format(float(param)))
        axes[1].plot(time, d_in['{}_sh'.format(var_name)][idx, :],
                     label='{:.1f}'.format(float(param)))

    axes[0].set_title('NH')
    axes[0].legend()
    axes[0].grid(b=True)

    axes[1].set_title('SH')
    axes[1].legend()
    axes[1].grid(b=True)

    plt.tight_layout()
    plt.savefig('plt_compare_{}_{}.png'.format(var_name, param_name))
    plt.close()

    nh_seas = d_in['{}_nh'.format(var_name)].groupby('time.season')
    sh_seas = d_in['{}_sh'.format(var_name)].groupby('time.season')
    nh_sm = nh_seas.mean(axis=1)
    sh_sm = sh_seas.mean(axis=1)
    nh_svar = nh_seas.std(axis=1)
    sh_svar = sh_seas.std(axis=1)

    fig, axis = plt.subplots(1, 2, figsize=(12, 5))
    cols_nh = [0, 3, 2, 1]
    cols_sh = [3, 0, 1, 2]
    for snx, season in enumerate(nh_sm.season):
        axis[0].plot(param_vals, nh_sm[:, snx], 'C{}o-'.format(cols_nh[snx]),
                     label=str(season.data))

        axis[1].plot(param_vals, sh_sm[:, snx], 'C{}o-'.format(cols_sh[snx]),
                     label=str(season.data))

    axis[0].set_xlabel(PARAMS[param_name])
    axis[0].set_ylabel('Mean Jet {name} [{units}]'.format(**VARS[var_name]))
    axis[0].legend()
    axis[0].set_title('Northern Hemisphere')
    axis[0].grid(b=True, ls='--')

    axis[1].set_xlabel(PARAMS[param_name])
    axis[1].legend()
    axis[1].set_title('Southern Hemisphere')
    axis[1].grid(b=True, ls='--')

    if var_name == 'lat':
        axis[0].set_ylim([25, 45])
        axis[1].invert_yaxis()
        axis[1].set_ylim([-45, -25])
    plt.suptitle('Seasonal Mean Jet {}'.format(VARS[var_name]['name']))
    #plt.tight_layout()
    plt.savefig('plt_season_mean_{}_{}.png'.format(var_name, param_name))
    #plt.show()

if __name__ == "__main__":
    PARAMS = {'fit': 'Polynomial Fit [deg]', 'y0': 'Minimum Latitude',
              'pv_lev': 'PV Level [PVU]'}
    VARS = {'lat': {'name': 'Latitude Position', 'units': 'deg'},
            'theta': {'name': 'Theta Position', 'units': 'K'},
            'intens': {'name': 'Intensity', 'units': 'm/s'}}
    for var_name in VARS:
        main('pv_lev', np.arange(1.0, 4.5, 0.5), var_name)
        #main('fit', np.arange(4, 9), var_name)
    #main('y0', np.arange(1, 11))
