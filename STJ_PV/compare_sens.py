# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as sts

#plt.style.use('ggplot')
NC_DIR = './jet_out'

def main(param_name, param_vals, dates, var_name='lat'):
    """
    Compare jet latitudes of results from sensitivity runs of stj_run.

    Parameters
    ----------
    param_name : string
        Name of sensitivity parameter
    param_vals : iterable
        Array or list of values that `param_name` takes

    """
    sens_out = {}
    opts = {'run_type': 'ERAI_MONTHLY_THETA_STJPV', 'fit': 8, 'y0': 10, 'pv_lev': 2.0}
    opts.pop(param_name)
    opts_var = [{param_name: p_val} for p_val in param_vals]
    file_fmt = '{run_type}_pv{pv_lev:.1f}_fit{fit:.0f}_y0{y0:03.1f}_{}_{}.nc'
    file_fmt = os.path.join(NC_DIR, file_fmt)
    files_in  = [file_fmt.format(*dates, **var, **opts) for var in opts_var]
    d_in = xr.concat([xr.open_dataset(in_f) for in_f in files_in], dim=param_name)
    d_in[param_name] = param_vals

    time = d_in.time

    # Figure size set to 129 mm wide, 152 mm tall
    fig_mult = 1.0
    fig_width = 129 * fig_mult
    fig_size = (fig_width / 25.4, (fig_width / 3 ) / 25.4)
    plt.rc('font', family='sans-serif', size=8 * fig_mult)

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
    plt.savefig('plt_compare_{}_{}.{}'.format(var_name, param_name, EXTN))
    plt.close()

    nh_seas = d_in['{}_nh'.format(var_name)].groupby('time.season')
    sh_seas = d_in['{}_sh'.format(var_name)].groupby('time.season')
    nh_sm = nh_seas.mean(axis=1)
    sh_sm = sh_seas.mean(axis=1)
    nh_svar = nh_seas.std(axis=1)
    sh_svar = sh_seas.std(axis=1)

    # Figure size set to 129 mm wide, 152 mm tall
    fig_mult = 1.0
    fig_width = 129 * fig_mult
    fig_size = (fig_width / 25.4, (fig_width * 0.6) / 25.4)
    plt.rc('font', family='sans-serif', size=8 * fig_mult)

    fig, axis = plt.subplots(1, 2, figsize=fig_size)
    cols_nh = ['0x', '3o', '2.', '1v']
    cols_sh = ['3o', '0x', '1v', '2.']
    for snx, season in enumerate(nh_sm.season):
        if param_name == 'fit':
            # Fit is discrete variable that can only be an integer, don't put a line in
            line_str = 'C{}'
        else:
            line_str = 'C{}-'
        axis[0].plot(param_vals, nh_sm[:, snx], line_str.format(cols_nh[snx]),
                     label=str(season.data))

        axis[1].plot(param_vals, sh_sm[:, snx], line_str.format(cols_sh[snx]),
                     label=str(season.data))
        sens_out[('NH', str(season.data))] = sts.linregress(param_vals, nh_sm[:, snx])
        sens_out[('SH', str(season.data))] = sts.linregress(param_vals, sh_sm[:, snx])

    grid_style = {'ls': '--', 'lw': 0.5}
    axis[0].set_xlabel(PARAMS[param_name])
    axis[0].set_ylabel('Mean Jet {name} [{units}]'.format(**VARS[var_name]))
    if param_name == 'fit' and var_name == 'lat':
        axis[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.5))
    else:
        axis[0].legend()
    axis[0].set_title('(a) Northern Hemisphere')
    axis[0].grid(b=True, **grid_style)
    axis[1].set_xlabel(PARAMS[param_name])
    axis[1].legend()
    axis[1].set_title('(b) Southern Hemisphere')
    axis[1].grid(b=True, **grid_style)

    if var_name == 'lat':
        axis[0].set_ylim([25, 45])
        #axis[1].invert_yaxis()
        #axis[1].set_ylim([-45, -25])
        axis[1].set_ylim([-25, -45])
    fig.subplots_adjust(left=0.11, bottom=0.13, right=0.97, top=0.87, wspace=0.26)
    plt.suptitle('Seasonal Mean Jet {}'.format(VARS[var_name]['name']))
    plt.savefig('plt_season_mean_{}_{}.{}'.format(var_name, param_name, EXTN))
    #plt.show()

    return sens_out


def sens_num(param_vals, data, names):
    """Get min/max and sensitivity of `data` as a function of `param_vals`."""
    var_name, param_name, season, hem = names

    sens = sts.linregress(param_vals, data)

    #print('Sensitivity of {} to {} in {} {}: {:.2f} (r: {:.3f}, p: {:.4f})'
    #      .format(var_name, param_name, hem, season, sens[0], sens[2], sens[3]))


if __name__ == "__main__":
    PARAMS = {'fit': 'Polynomial Fit [deg]', 'y0': 'Minimum Latitude',
              'pv_lev': 'PV Level [PVU]'}
    VARS = {'lat': {'name': 'Latitude Position', 'units': 'deg'},
            'theta': {'name': 'Theta Position', 'units': 'K'},
            'intens': {'name': 'Intensity', 'units': 'm/s'}}
    EXTN = 'eps'
    dates = ('1979-01-01', '2016-12-31')
    sens = {'pv': {}, 'fit': {}, 'y0': {}}
    for var_name in VARS:
        sens['pv'][var_name] = main('pv_lev', np.arange(1.0, 4.5, 0.5), dates, var_name)
        sens['fit'][var_name] = main('fit', np.arange(5, 9), dates, var_name)
        sens['y0'][var_name] = main('y0', np.arange(2.5, 15, 2.5), dates, var_name)

    for hem in ['NH', 'SH']:
        for var in sens:
            out_line = '{} & {} & '.format(hem, var)
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                if sens[var]['lat'][(hem, season)].pvalue <= 0.05:
                    fmt_str = ' \\textbf{%.4f} & '
                else:
                    fmt_str = ' %.4f & '
                out_line += fmt_str % sens[var]['lat'][(hem, season)].slope
            print(out_line)
    #main('y0', np.arange(1, 11))
