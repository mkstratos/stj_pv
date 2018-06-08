# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns

SEASONS = np.array([None, 'DJF', 'DJF', 'MAM', 'MAM', 'MAM',
                    'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF'])

HEMS = {'nh': 'Northern Hemisphere', 'sh': 'Southern Hemisphere'}

class FileDiag(object):
    """
    Contains information about an STJ metric output file in a DataFrame.
    """
    def __init__(self, info, opts_hem=None, file_path=None):
        self.name = info['label']
        if file_path is None:
            # If the file path is not provided, the input path in `info` is the abs path
            file_path = ''
        self.d_s = xr.open_dataset(os.path.join(file_path, info['file']))

        self.dframe = None
        self.vars = None
        self.opt_hems = opts_hem

        var, self.start_t, self.end_t = self.make_dframe()
        self.metric = var

    def make_dframe(self):
        """Creates dataframe from input netCDF / xarray."""
        if self.opt_hems is None:
            hems = ['nh', 'sh']
        else:
            # in case you want to use equator or only one hemi
            hems = self.opt_hems

        self.dframe = self.d_s.to_dataframe()

        self.vars = set([var.split('_')[0] for var in self.dframe])
        if 'time' in self.vars:
            self.vars.remove('time')
        dframes = [[pd.DataFrame({var: self.dframe['{}_{}'.format(var, hem)], 'hem': hem})
                    for var in self.vars] for hem in hems]
        dframes_tmp = []
        for frames in dframes:
            metric_hem = None
            for frame in frames:
                # Add a time column so that the merge works
                frame['time'] = frame.index
                if metric_hem is None:
                    metric_hem = frame
                else:
                    metric_hem = metric_hem.merge(frame)

            dframes_tmp.append(metric_hem)
        metric = dframes_tmp[0].append(dframes_tmp[1])

        if len(hems) == 3:  # If eq is also wanted
            metric = metric.append(dframes_tmp[2])
        metric['season'] = SEASONS[pd.DatetimeIndex(metric.time).month].astype(str)
        metric['kind'] = self.name

        # Make all times have Hour == 0
        times = pd.DatetimeIndex(metric['time'])
        if all(times.hour == times[0].hour):
            times -= pd.Timedelta(hours=times[0].hour)

        metric.index = times

        return metric, metric.index[0], metric.index[-1]

    def append_metric(self, other):
        """Append the DataFrame attribute (self.lats) to another FileDiag's DataFrame."""
        assert isinstance(other, FileDiag)
        return self.metric.append(other.metric)

    def time_subset(self, other):
        """Make two fds objects have matching times."""
        if self.metric.shape[0] > other.metric.shape[0]:
            # self is bigger
            fds0 = self.metric.loc[sorted(other.metric.time[other.metric.hem == 'nh'])]
            self.metric = fds0
            self.start_t = self.metric.index[0]
            self.end_t = self.metric.index[-1]
        else:
            # other is bigger
            fds1 = other.metric.loc[sorted(self.metric.time[self.metric.hem == 'nh'])]
            other.metric = fds1
            other.start_t = other.metric.index[0]
            other.end_t = other.metric.index[-1]

    def __sub__(self, other):
        hems = ['nh', 'sh']

        df1 = self.metric
        df2 = other.metric
        assert (df1.time - df2.time).sum() == pd.Timedelta(0), 'Not all times match'

        # Get a set of all variables common to both datasets
        var_names = self.vars.intersection(other.vars)

        # Initialise a list of differences of the variables between datasets
        diff = []
        for var in var_names:
            # Separate hemispheres for `var` one list for self, one for other
            inside = [df1[df1.hem == hem][var] for hem in hems]
            outside = [df2[df2.hem == hem][var] for hem in hems]

            # For each hemisphere, make the difference of self - other a DataFrame
            diff_c = [pd.DataFrame({var: inside[idx] - outside[idx], 'hem': hems[idx],
                                    'time': df1[df1.hem == hems[idx]].time})
                      for idx in range(len(hems))]

            # Combine the two hemispheres into one DF
            diff.append(diff_c[0].append(diff_c[1]))

            diff_out = None
            for frame in diff:
                if diff_out is None:
                    diff_out = frame
                else:
                    diff_out = diff_out.merge(frame)

        diff_out['season'] = SEASONS[pd.DatetimeIndex(diff_out.time).month].astype(str)
        return diff_out


def main():
    """Selects two files to compare, loads and plots them."""
    file_info = {'NCEP-mon':
                 {'file': ('NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'NCEP Monthly'},

                 'NCEP-mon-70max':
                 {'file': ('NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit8_y010.0_yN70.0_zmean_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'NCEP Monthly 70Max'},

                 'NCEP-day':
                 {'file': ('NCEP_NCAR_DAILY_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'NCEP Daily Z-Median'},

                 'NCEP-day-zmean':
                 {'file': ('NCEP_NCAR_DAILY_STJPV_pv2.0_fit8_y010.0_zmean_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'NCEP Daily Z-Mean'},

                 'NCEP-PV': {'file': 'NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit12_y010.0.nc',
                             'label': 'NCEP PV'},

                 'NCEP-Umax': {'file': ('NCEP_NCAR_MONTHLY_HR_STJUMax_pres25000.0'
                                        '_y010.0.nc'), 'label': 'NCEP U-max'},

                 'ERAI-Theta':
                 {'file': ('ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'Monthly ERAI PV'},

                 'ERAI-Theta-5':
                 {'file': ('ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y05.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'),
                  'label': 'B Monthly ERAI PV 5.0˚'},

                 'ERAI-Theta-DM':
                 {'file': ('ERAI_MONTHLY_DM_THETA_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'),
                  'label': 'A Monthly mean of daily ERAI PV'},

                 'ERAI-Theta-Day':
                 {'file': ('ERAI_DAILY_THETA_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'Daily ERAI PV median'},

                 'ERAI-Theta-Day_zmean':
                 {'file': ('ERAI_DAILY_THETA_STJPV_pv2.0_fit8_y010.0_zmean_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'Daily ERAI PV mean'},

                 'ERAI-Theta-Day-5':
                 {'file': ('ERAI_DAILY_THETA_STJPV_pv2.0_fit8_y05.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'B Daily ERAI PV 5.0˚'},

                 'ERAI-Regrid':
                 {'file': ('ERAI_MONTHLY_THETA_2p5_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'ERAI Theta 2.5'},

                 'ERAI-Uwind':
                 {'file': 'ERAI_PRES_STJUMax_pres25000.0_y010.0_1979-01-01_2016-12-31.nc',
                  'label': 'ERAI U-Wind'},

                 'ERAI-Theta5': {'file': 'ERAI_MONTHLY_THETA_STJPV_pv2.0_fit5_y010.0.nc',
                                 'label': 'ERAI Theta5'},

                 'ERAI-Pres':
                 {'file': ('ERAI_PRES_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2015-12-31.nc'), 'label': 'ERAI Pres'},

                 'ERAI-Pres-newlev':
                 {'file': ('ERAI_PRES_STJPV_pv2.0_fit8_y010.0_zmean_'
                           '1979-01-01_2016-12-31_newlevels.nc'),
                  'label': 'ERAI Pres new'},

                 'ERAI-Pres-oldlev':
                 {'file': ('ERAI_PRES_STJPV_pv2.0_fit8_y010.0_zmean_'
                           '1979-01-01_2016-12-31_oldlevels.nc'),
                  'label': 'ERAI Pres old'},

                 'ERAI-Epv':
                 {'file': ('ERAI_EPVPRES_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2015-12-31.nc'),
                  'label': 'ERAI EPV Pres'},

                 'ERAI-KP': {'file': ('ERAI_PRES_KangPolvani_zmedian_'
                                      '1979-01-01_2015-12-31.nc'), 'label': 'ERAI K-P'},

                 'ERAI-Theta_LR':
                 {'file': ('ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0_zmedian_lon45-100_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'Monthly ERAI PV Slice'},

                 'ERAI-Theta-Day_LR':
                 {'file': ('ERAI_DAILY_THETA_STJPV_pv2.0_fit8_y010.0_zmedian_lon45-100_'
                           '1979-01-01_2016-12-31.nc'), 'label': 'A Daily ERAI PV'},

                 'MERRA-Mon':
                 {'file': ('MERRA_MONTHLY_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2015-12-31.nc'), 'label': 'Monthly MERRA PV'},
                 'JRA-Mon':
                 {'file': ('JRA55_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0_zmedian_'
                           '1979-01-01_2017-12-31.nc'), 'label': 'Monthly JRA-55 PV'},

                 'ERAI-Theta_zmean':
                 {'file': ('ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0_zmean_'
                           '1979-01-01_2016-12-31.nc'),
                  'label': 'Monthly ERAI PV Zonal Mean'}}

    nc_dir = './jet_out'
    if not os.path.exists(nc_dir):
        nc_dir = '.'

    fig_mult = 2.0
    plt.rc('font', size=9 * fig_mult)
    extn = 'png'
    sns.set_style('whitegrid')
    fig_width = (9.5 / 2.54) * fig_mult
    fig_height = (11.5 / 2.54) * fig_mult

    in_names = ['NCEP-mon', 'NCEP-mon-70max']

    fds = [FileDiag(file_info[in_name], file_path=nc_dir) for in_name in in_names]

    data = fds[0].append_metric(fds[1])

    # Make violin plot grouped by hemisphere, then season
    # NOTE: I've changed the seaborn.violinplot code to make the quartile lines
    # solid rather than dashed, may want to come back to this and figure out a way
    # to implement it in a nice (non-hacked!) way for others and PR it to seaborn
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    sns.violinplot(x='season', y='lat', hue='kind', data=data[data.hem == 'nh'],
                   split=True, inner='quart', ax=axes[0], cut=0, linewidth=1.0 * fig_mult,
                   dashpattern='-')
    axes[0].set_yticks(np.arange(30, 60, 10))

    sns.violinplot(x='season', y='lat', hue='kind', data=data[data.hem == 'sh'],
                   split=True, inner='quart', ax=axes[1], cut=0, linewidth=1.0 * fig_mult,
                   dashpattern='-')
    axes[1].set_yticks(np.arange(-50, -20, 10))
    fig.subplots_adjust(left=0.10, bottom=0.08, right=0.95, top=0.94, hspace=0.0)
    fig.legend(bbox_to_anchor=(0.15, 0.94), loc='upper left', borderaxespad=0.)

    for axis in axes:
        axis.legend_.remove()
        axis.tick_params(axis='y', rotation=90)
        axis.grid(b=True, ls='--', zorder=-1)
    fig.suptitle('Seasonal jet latitude distributions')

    plt.savefig('plt_dist_{}-{}.{ext}'.format(ext=extn, *in_names))
    plt.close()

    if fds[0].start_t != fds[1].start_t or fds[0].end_t != fds[1].end_t:
        fds[0].time_subset(fds[1])
    diff = fds[0] - fds[1]

    # Make timeseries plot for each hemisphere, and difference in each
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    for idx, dfh in enumerate(data.groupby('hem')):
        hem = dfh[0]
        axes[idx, 1].plot(diff.lat[diff.hem == hem])

        for kind, dfk in dfh[1].groupby('kind'):
            axes[idx, 0].plot(dfk.lat, label=kind)
        axes[idx, 0].set_title(HEMS[hem])
        axes[idx, 1].set_title('{} Difference'.format(HEMS[hem]))
        axes[idx, 0].grid(b=True, ls='--')
        axes[idx, 1].grid(b=True, ls='--')

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig('plt_diff_timeseries_{}-{}.{ext}'.format(ext=extn, *in_names))

    # Make a bar chart of mean difference
    sns.factorplot(x='season', y='lat', col='hem', data=diff, kind='bar')
    plt.tight_layout()
    plt.savefig('plt_diff_bar_{}-{}.{ext}'.format(ext=extn, *in_names))


if __name__ == "__main__":
    main()
