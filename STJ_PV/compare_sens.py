# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as sts

NC_DIR = './jet_out/sens'


def main(run_type, param_name, param_vals, dates, var='lat'):
    """
    Compare jet latitudes of results from sensitivity runs of stj_run.

    Parameters
    ----------
    param_name : string
        Name of sensitivity parameter
    param_vals : iterable
        Array or list of values that `param_name` takes

    """
    # opts = {'run_type': 'ERAI_MONTHLY_THETA_STJPV',
    #         'fit': 8, 'y0': 10, 'yN': 70, 'pv_lev': 2.0}
    opts = {'run_type': run_type, 'fit': 6,
            'y0': 10, 'pv_lev': 2.0, 'yN': 90.0}
    opts.pop(param_name)
    opts_var = [{param_name: p_val} for p_val in param_vals]
    file_fmt = ('{run_type}_pv{pv_lev:.1f}_fit{fit:.0f}_'
                'y0{y0:03.1f}_yN{yN:.1f}_zmean_{}_{}.nc')
    file_fmt = os.path.join(NC_DIR, file_fmt)
    files_in = [file_fmt.format(*dates, **var, **opts) for var in opts_var]
    d_in = xr.concat([xr.open_dataset(in_f)
                      for in_f in files_in], dim=param_name)
    d_in[param_name] = param_vals

    # Figure size set to 129 mm wide, 152 mm tall
    fig_mult = 2.0
    fig_width = 129 * fig_mult
    plt.rc('font', family='sans-serif', size=8 * fig_mult)

    fig_size = (fig_width / 25.4, (fig_width * 0.6) / 25.4)
    figure = plt.subplots(2, 2, figsize=fig_size, sharex=True)
    sens_monthly(d_in, var, param_name, figure)
    plt.close()

    fig_size = (fig_width / 25.4, (fig_width * 0.6) / 25.4)
    figure = plt.subplots(1, 2, figsize=fig_size)
    sens_out = sens_seasonal(d_in, var, param_name, figure)
    plt.close()

    fig_size = (fig_width / 25.4, (fig_width / 3) / 25.4)
    figure = plt.subplots(2, 1, figsize=fig_size)
    plot_timeseries(d_in, var, param_name, figure)
    plt.close()

    return sens_out


def plot_timeseries(d_in, var, param_name, figure):
    """Plot timeseries of for all parameter values."""
    # Figure size set to 129 mm wide, 152 mm tall
    fig, axes = figure
    axes = axes.ravel()

    for idx, param in enumerate(d_in[param_name]):
        axes[0].plot(d_in.time, d_in['{}_nh'.format(var)][idx, :],
                     label='{:.1f}'.format(float(param)))
        axes[1].plot(d_in.time, d_in['{}_sh'.format(var)][idx, :],
                     label='{:.1f}'.format(float(param)))

    axes[0].set_title('NH')
    axes[0].legend()
    axes[0].grid(**GRID_STYLE)

    axes[1].set_title('SH')
    axes[1].legend()
    axes[1].grid(**GRID_STYLE)

    plt.tight_layout()
    fig.savefig('plt_compare_{}_{}.{}'.format(var, param_name, EXTN))


def sens_seasonal(d_in, var, param_name, figure):
    """Plot seasonal sensitivity to a parameter."""
    sens_out = {}
    fig, axis = figure
    cols_nh = ['0x', '3o', '2.', '1v']
    cols_sh = ['3o', '0x', '1v', '2.']
    param_vals = d_in[param_name]

    nh_seas = d_in['{}_nh'.format(var)].groupby('time.season')
    sh_seas = d_in['{}_sh'.format(var)].groupby('time.season')
    nh_sm = nh_seas.mean(axis=1)
    sh_sm = sh_seas.mean(axis=1)

    for snx, season in enumerate(nh_sm.season):
        if param_name == 'fit':
            # Degree of fit is discrete variable that can only be
            # an integer, don't put a line in
            line_str = 'C{}'
        else:
            line_str = 'C{}-'
        axis[0].plot(param_vals, nh_sm[:, snx], line_str.format(cols_nh[snx]),
                     label=str(season.data))

        axis[1].plot(param_vals, sh_sm[:, snx], line_str.format(cols_sh[snx]),
                     label=str(season.data))
        sens_out[('NH', str(season.data))] = sts.linregress(param_vals,
                                                            nh_sm[:, snx])
        sens_out[('SH', str(season.data))] = sts.linregress(param_vals,
                                                            sh_sm[:, snx])

    axis[0].set_xlabel(PARAMS[param_name])
    axis[0].set_ylabel('Mean Jet {name} [{units}]'.format(**VARS[var]))
    if param_name == 'fit' and var == 'lat':
        axis[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.5))
    else:
        axis[0].legend()
    axis[0].set_title('(a) Northern Hemisphere')
    axis[0].grid(**GRID_STYLE)
    axis[1].set_xlabel(PARAMS[param_name])
    axis[1].legend()
    axis[1].set_title('(b) Southern Hemisphere')
    axis[1].grid(**GRID_STYLE)

    if var == 'lat':
        axis[0].set_ylim([25, 45])
        # axis[1].invert_yaxis()
        # axis[1].set_ylim([-45, -25])
        axis[1].set_ylim([-25, -45])
    fig.subplots_adjust(left=0.11, bottom=0.13, right=0.97,
                        top=0.87, wspace=0.26)
    plt.suptitle('Seasonal Mean Jet {}'.format(VARS[var]['name']))
    fig.savefig('plt_season_mean_{}_{}.{}'.format(var, param_name, EXTN))
    return sens_out


def sens_monthly(data_in, var, param_name, figure):
    """Plot monthly mean and variance for different values of parameters."""
    fig, axes = figure
    nh_monthly = data_in['{}_nh'.format(var)].groupby('time.month')
    sh_monthly = data_in['{}_sh'.format(var)].groupby('time.month')

    nh_mm = nh_monthly.mean(axis=1)
    sh_mm = sh_monthly.mean(axis=1)

    nh_mvar = nh_monthly.std(axis=1)
    sh_mvar = sh_monthly.std(axis=1)

    nh_mm.plot.line(x='month', hue=param_name,
                    ax=axes[0, 0], add_legend=True)
    nh_mvar.plot.line(x='month', hue=param_name,
                      ax=axes[0, 1], add_legend=False)
    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    label = '{name} [{units}]'.format(**VARS[var])
    axes[0, 0].set_title('NH Mean {}'.format(label))
    axes[0, 1].set_title('NH Std {}'.format(label))

    sh_mm.plot.line(x='month', hue=param_name,
                    ax=axes[1, 0], add_legend=False)
    sh_mvar.plot.line(x='month', hue=param_name,
                      ax=axes[1, 1], add_legend=False)
    axes[1, 0].set_title('SH Mean {}'.format(label))
    axes[1, 1].set_title('SH Std {}'.format(label))
    for axis in axes:
        axis[0].grid(**GRID_STYLE)
        axis[1].grid(**GRID_STYLE)
        axis[0].set_ylabel(VARS[var]['units'])
        axis[1].set_ylabel(VARS[var]['units'])

    plt.tight_layout()
    plt.savefig('plt_sens_{}_{}_monthly.{}'.format(var, param_name, EXTN))


def run():
    """Set dates, variable names, and parameters to plot sensitivity."""
    # run_type = 'NCEP_NCAR_MONTHLY_STJPV'
    run_type = 'ERAI_MONTHLY_THETA_STJPV'
    dates = ('1979-01-01', '2016-12-31')
    param_vals = {'pv_lev': np.arange(1.0, 4.5, 0.5),
                  'fit': np.arange(5, 9),
                  'y0': np.arange(2.5, 15, 2.5),
                  'yN': np.arange(60., 95., 5.)}
    # sens = {'pv': {}, 'fit': {}, 'y0': {}, 'yN': {}}
    sens = {'yN': {}}

    for var_name in VARS:
        for param in sens:
            sens[param][var_name] = main(run_type, param, param_vals[param],
                                         dates, var_name)
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


# Global static information
PARAMS = {'fit': 'Polynomial Fit [deg]', 'y0': 'Minimum Latitude',
          'pv_lev': 'PV Level [PVU]', 'yN': 'Maximum Latitude'}
VARS = {'lat': {'name': 'Latitude Position', 'units': 'deg'},
        'theta': {'name': 'Theta Position', 'units': 'K'},
        'intens': {'name': 'Intensity', 'units': 'm/s'}}
EXTN = 'png'
GRID_STYLE = {'b': True, 'ls': '--', 'lw': 0.5}

if __name__ == "__main__":
    run()
