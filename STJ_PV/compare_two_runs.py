# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns
import pdb

SEASONS = np.array([None, 'DJF', 'DJF', 'MAM', 'MAM', 'MAM',
                    'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF'])

HEMS = {'nh': 'Northern Hemisphere', 'sh': 'Southern Hemisphere'}

class FileDiag(object):
    """
    Contains information about an STJ metric output file in a DataFrame.
    """
    def __init__(self, info, var_name):
        self.name = info['label']
        self.d_s = xr.open_dataset(info['file'])
        self.dframe = None
        self.var_name = var_name
        var, self.start_t, self.end_t = self.make_dframe()
        setattr(self, 'metric', var) 

    def make_dframe(self):
        """Creates dataframe from input netCDF / xarray."""
        hems = ['nh', 'sh']
        self.dframe = self.d_s.to_dataframe()


        var = [pd.DataFrame({self.var_name: self.dframe['{}_{}'.format(self.var_name, hem)], 'hem': hem})
                for hem in hems]
        var = var[0].append(var[1])
        var['season'] = SEASONS[var.index.month].astype(str)
        var['kind'] = self.name

        return var, self.dframe.index[0], self.dframe.index[-1]

    def append(self, other):
        """Append the DataFrame attribute (self.lats) to another FileDiag's DataFrame."""
        assert isinstance(other, FileDiag)
        
        return self.metric.append(other.metric)
        
    def __sub__(self, other): 
        hems = ['nh', 'sh']

        df1 = getattr(self, 'metric')
        df2 = getattr(other, 'metric')

        diff = [pd.DataFrame({self.var_name: getattr(df1[getattr(df1,'hem') == hem], self.var_name)-
                                             getattr(df2[getattr(df2,'hem') == hem], other.var_name),
                              'hem': hem}) for hem in hems]

        diff = diff[0].append(diff[1])
        diff['season'] = SEASONS[diff.index.month].astype(str)
        return diff


def main():
    """Selects two files to compare, loads and plots them."""
    file_info = {
        'NCEP-PV': {'file': './NCEP_NCAR_MONTHLY_STJPV_pv2.0_fit12_y010.0.nc',
                    'label': 'NCEP PV'},
        'NCEP-Umax': {'file': './NCEP_NCAR_MONTHLY_HR_STJUMax_pres25000.0_y010.0.nc',
                      'label': 'NCEP U-max'},
        'ERAI-Theta': {'file': './ERAI_MONTHLY_THETA_STJPV_pv2.0_fit8_y010.0.nc',
                       'label': 'ERAI Theta'},
        'ERAI-Pres': {'file': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc',
                      'label': 'ERAI PV'},
        'ERAI-KP': {'file': './ERAI_PRES_KangPolvani_1979-01-01_2016-01-01.nc',
                    'label': 'ERAI K-P'}
    }

    fig_width = 110 / 25.4
    in_names = ['NCEP-PV', 'NCEP-Umax']
    fds = [FileDiag(file_info[in_name], 'metric') for in_name in in_names]

    assert fds[0].start_t == fds[1].start_t  , 'Start dates are different'
    assert fds[0].end_t   == fds[1].end_t  , 'Start dates are different'

    data = fds[0].append(fds[1])
    diff = fds[0] - fds[1]

    # Make violin plot grouped by hemisphere, then season
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_width * 2))
    sns.violinplot(x='season', y='lat', hue='kind', data=data[data.hem == 'nh'],
                   split=True, inner='quart', ax=axes[0], cut=0)
    sns.violinplot(x='season', y='lat', hue='kind', data=data[data.hem == 'sh'],
                   split=True, inner='quart', ax=axes[1], cut=0)
    fig.legend()
    for axis in axes:
        axis.legend_.remove()

    plt.savefig('plt_dist.png')
    plt.close()


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
    plt.savefig('plt_diff_timeseries.png')

    # Make a bar chart of mean difference
    sns.factorplot(x='season', y='lat', col='hem', data=diff, kind='bar')
    plt.tight_layout()
    plt.savefig('plt_diff_bar.png')


if __name__ == "__main__":
    main()
