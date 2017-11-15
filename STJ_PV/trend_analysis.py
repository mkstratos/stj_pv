# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.stats import mstats
import matplotlib.lines as mlines
import pdb

plt.style.use('ggplot')

def get_linear_trend(x, y, time_dim):
    slope, intercept, r_value, p_value, std_err = mstats.linregress(x,y)

    if p_value > 0.05:
        print '    Not significant: {:.3f} with mean {:.2f}'.format(p_value, np.mean(y.data))
    else:
        print '    Is  significant: {:.3f} with mean {:.2f}'.format(p_value, np.mean(y.data)) 

    print '        Trend: total {:.3f} or {:.3f}/yr'.format(slope*len(y), slope*len(y)/time_dim)

    return slope, intercept

def plot_dependencies(data_nh, data_sh):

    time_array = np.arange(0,len(data_nh))
    num_years = len(data_nh)/12. 
    season_array = np.arange(0,len(data_nh)/4)

    nh_sm = data_nh.groupby('time.season')
    sh_sm = data_sh.groupby('time.season')

    return time_array, num_years, season_array, nh_sm, sh_sm

def make_plot(data_nh, data_sh, dates, flag):

    if flag == 'lat':
        filename = 'plt_season_timeseries.png'
        min_range, max_range = 20, 55
        min_range_sh, max_range_sh = -20, -55
    else:
        filename = 'plt_season_timeseries_int.png'
        min_range, max_range = 10, 45
        min_range_sh, max_range_sh = 10, 45

    time_array, num_years, season_array, nh_sm, sh_sm = plot_dependencies(data_nh, data_sh)

    print 'nh timeseries'
    slope_nh, intercept_nh = get_linear_trend(time_array, data_nh, num_years)
    print 'sh timeseries'
    slope_sh, intercept_sh = get_linear_trend(time_array, data_sh,  num_years)


    fig, axis = plt.subplots(3, 2, figsize=(20, 6))
    
    axis[2,0].plot(dates[0], data_nh, '-', c='k')
    axis[2,1].plot(dates[0], data_sh, '-', c='k', label='Time series')
    axis[2,0].plot([dates[0][0],dates[0][-1]], 
                   [intercept_nh, slope_nh* season_array[-1]+intercept_nh], c='k')
    axis[2,1].plot([dates[0][0],dates[0][-1]], 
                   [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='k')
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        print season
        nh_ts = data_nh[sh_sm.groups[season]]
        sh_ts = data_sh[sh_sm.groups[season]]
        dates_season = dates[0][sh_sm.groups[season]]

        print 'nh'
        slope_nh, intercept_nh = get_linear_trend(season_array, nh_ts, num_years)
        print 'sh'
        slope_sh, intercept_sh = get_linear_trend(season_array, sh_ts, num_years)

        if season == 'DJF': 
            axis[0,0].plot(dates_season, nh_ts, 'x', c='b',label='Winter')
            axis[0,1].plot(dates_season, sh_ts, 'x', c='r',label='Summer')
            axis[0,0].plot([dates_season[0],dates_season[-1]], 
                           [intercept_nh, slope_nh* season_array[-1]+intercept_nh], c='b')
            axis[0,1].plot([dates_season[0],dates_season[-1]], 
                           [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='r')

        elif season == 'JJA':
            axis[0,0].plot(dates_season, nh_ts, 'x',c='r', label='Summer')
            axis[0,1].plot(dates_season, sh_ts, 'x',c='b', label='Winter')
            axis[0,0].plot([dates_season[0],dates_season[-1]], 
                           [intercept_nh, slope_nh* season_array[-1]+intercept_nh] , c='r')
            axis[0,1].plot([dates_season[0],dates_season[-1]], 
                           [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='b')

        elif season == 'SON':
            axis[1,0].plot(dates_season, nh_ts, 'x', c ='orange',label='Autumn')
            axis[1,1].plot(dates_season, sh_ts, 'x', c='g', label='Spring')
            axis[1,0].plot([dates_season[0],dates_season[-1]], 
                           [intercept_nh, slope_nh* season_array[-1]+intercept_nh], c='orange')
            axis[1,1].plot([dates_season[0],dates_season[-1]], 
                           [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='g')

        else:
            axis[1,0].plot(dates_season, nh_ts, 'x',c='g', label='Spring')
            axis[1,1].plot(dates_season, sh_ts, 'x', c='orange',label='Autumn')
            axis[1,0].plot([dates_season[0],dates_season[-1]], 
                           [intercept_nh, slope_nh* season_array[-1]+intercept_nh] , c='g')
            axis[1,1].plot([dates_season[0],dates_season[-1]], 
                           [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='orange')


    #for j, row in enumerate (axis):
    #    for i, ax in enumerate (row):
    for j in xrange(2):
            if j == 0:
                #NH
                axis[0,j].set_ylim([min_range, max_range]) 
                axis[1,j].set_ylim([min_range, max_range]) 
                axis[2,j].set_ylim([min_range, max_range]) 
            else:
                #SH
                axis[0,j].set_ylim([min_range_sh,max_range_sh]) 
                axis[1,j].set_ylim([min_range_sh,max_range_sh]) 
                axis[2,j].set_ylim([min_range_sh,max_range_sh]) 

    axis[0,0].set_title('Northern Hemisphere')
    axis[0,1].set_title('Southern Hemisphere')

    summer = mlines.Line2D([], [], color='red', marker='x',
                           markersize=5,linestyle = '',label='Summer')
    winter = mlines.Line2D([], [], color='blue', marker='x',
                           markersize=5,linestyle = '',label='Winter')
    autumn = mlines.Line2D([], [], color='orange', marker='x',
                           markersize=5,linestyle = '',label='Autumn')
    spring = mlines.Line2D([], [], color='green', marker='x',
                           markersize=5,linestyle = '',label='Spring')
    timeseries = mlines.Line2D([], [], color='black', 
                               linestyle = '-',label='Time series')

    l1 = plt.legend(handles = [winter, autumn, summer, spring, timeseries],
                    numpoints=1,loc=(1.05, 0.6))
    plt.gca().add_artist(l1) 

    plt.savefig(filename)
    plt.show()
    pdb.set_trace()


def main():


    files_in = {'Theta': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc'}
    ftypes = sorted(files_in.keys())
    d_in = {in_f: nc.Dataset(files_in[in_f], 'r') for in_f in files_in}

    times = [d_in[ftype].variables['time'] for ftype in ftypes]
    dates = [pd.DatetimeIndex(nc.num2date(time[:], time.units)) for time in times]


    lat_nh = {in_f: d_in[in_f].variables['lat_nh'][:] for in_f in d_in}['Theta']

    lat_sh = {in_f: d_in[in_f].variables['lat_sh'][:] for in_f in d_in}['Theta']
    int_nh = {in_f: d_in[in_f].variables['intens_nh'][:] for in_f in d_in}['Theta']
    int_sh = {in_f: d_in[in_f].variables['intens_sh'][:] for in_f in d_in}['Theta']

    lat_nh = xr.DataArray(lat_nh, coords=(dates[0],), dims=('time'))
    lat_sh = xr.DataArray(lat_sh, coords=(dates[0],), dims=('time'))
    
    int_nh = xr.DataArray(int_nh, coords=(dates[0],), dims=('time'))
    int_sh = xr.DataArray(int_sh, coords=(dates[0],), dims=('time'))

    make_plot(lat_nh, lat_sh, dates, 'lat')
    make_plot(int_nh, int_sh, dates, 'int')


if __name__ == "__main__":
    main()
