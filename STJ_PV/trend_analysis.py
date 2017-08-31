# -*- coding: utf-8 -*-
"""Script to compare two or more runs of STJ Find."""
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.stats import mstats
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

def main():


    files_in = {'Theta': './ERAI_PRES_STJPV_pv2.0_fit10_y010.0.nc'}
    ftypes = sorted(files_in.keys())
    d_in = {in_f: nc.Dataset(files_in[in_f], 'r') for in_f in files_in}
    time = d_in[ftypes[0]].variables['time']
    date = pd.DatetimeIndex(nc.num2date(time[:], time.units))
    lat_nh = {in_f: d_in[in_f].variables['lat_nh'][:] for in_f in d_in}['Theta']
    lat_sh = {in_f: d_in[in_f].variables['lat_sh'][:] for in_f in d_in}['Theta']

    lat_nh = xr.DataArray(lat_nh, coords=(date,), dims=('time'))
    lat_sh = xr.DataArray(lat_sh, coords=(date,), dims=('time'))

    time_array = np.arange(0,len(lat_nh))
    num_years = len(lat_nh)/12. 
    print 'nh timeseries'
    slope_nh, intercept_nh = get_linear_trend(time_array, lat_nh, num_years)
    print 'sh timeseries'
    slope_sh, intercept_sh = get_linear_trend(time_array, lat_sh,  num_years)

    nh_sm = lat_nh.groupby('time.season')
    sh_sm = lat_sh.groupby('time.season')

    fig, axis = plt.subplots(3, 2, figsize=(20, 6))
    #0th plot is winter, 1st is summer, 2nd is shoulders
    
    season_array = np.arange(0,len(lat_nh)/4)
    axis[2,0].plot(time_array, lat_nh, '-',label='Time series', c='k')
    axis[2,1].plot(time_array, lat_sh, '-',label='Time series', c='k')
    axis[2,0].plot([0,len(time_array)], [intercept_nh, slope_nh* season_array[-1]+intercept_nh], c='k')
    axis[2,1].plot([0,len(time_array)], [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='k')
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        print season
        nh_ts = lat_nh[sh_sm.groups[season]]
        sh_ts = lat_sh[sh_sm.groups[season]]
        print 'nh'
        slope_nh, intercept_nh = get_linear_trend(season_array, nh_ts, num_years)
        print 'sh'
        slope_sh, intercept_sh = get_linear_trend(season_array, sh_ts, num_years)

        #nh_mean = np.mean(nh_ts.data)
        #sh_mean = np.mean(sh_ts.data)

        if season == 'DJF':
            axis[0,0].plot(season_array, nh_ts, 'x-', c='b',label='Winter')
            axis[0,1].plot(season_array, sh_ts, 'x-', c='r',label='Summer')
            axis[0,0].plot([0,len(season_array)], [intercept_nh, slope_nh* season_array[-1]+intercept_nh], c='b')
            axis[0,1].plot([0,len(season_array)], [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='r')
        elif season == 'JJA':
            axis[0,0].plot(season_array, nh_ts, 'x-',c='r', label='Summer')
            axis[0,1].plot(season_array, sh_ts, 'x-',c='b', label='Winter')
            axis[0,0].plot([0,len(season_array)], [intercept_nh, slope_nh* season_array[-1]+intercept_nh] , c='r')
            axis[0,1].plot([0,len(season_array)], [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='b')

        elif season == 'SON':
            axis[1,0].plot(season_array, nh_ts, 'x-', c ='orange',label='Autumn')
            axis[1,1].plot(season_array, sh_ts, 'x-', c='g', label='Spring')
            axis[1,0].plot([0,len(season_array)], [intercept_nh, slope_nh* season_array[-1]+intercept_nh], c='orange')
            axis[1,1].plot([0,len(season_array)], [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='g')

        else:
            axis[1,0].plot(season_array, nh_ts, 'x-',c='g', label='Spring')
            axis[1,1].plot(season_array, sh_ts, 'x-', c='orange',label='Autumn')
            axis[1,0].plot([0,len(season_array)], [intercept_nh, slope_nh* season_array[-1]+intercept_nh] , c='g')
            axis[1,1].plot([0,len(season_array)], [intercept_sh, slope_sh* season_array[-1]+intercept_sh], c='orange')


    #for j, row in enumerate (axis):
    #    for i, ax in enumerate (row):
    for j in xrange(2):
            if j == 0:
                #NH
                axis[0,j].set_ylim([20, 50]) 
                axis[1,j].set_ylim([20, 50]) 
                axis[2,j].set_ylim([20, 50]) 
#                axis[3,j].set_ylim([20, 50]) 
            else:
                #SH
                axis[0,j].set_ylim([-20,-50]) 
                axis[1,j].set_ylim([-20,-50]) 
                axis[2,j].set_ylim([-20,-50]) 
#                axis[3,j].set_ylim([-20,-50]) 

    axis[2,0].set_xlabel('Time')
    axis[0,0].set_ylabel('Mean Jet Latitude')
    axis[0,0].set_title('Northern Hemisphere')
    axis[2,1].set_xlabel('Time')
    axis[0,1].set_title('Southern Hemisphere')
    #axis[0,0].set_ylabel('Winter')
    #axis[1,0].set_ylabel('Summer')
    #axis[1,0].set_ylabel('Shoulder')
    axis[0,1].legend(ncol=2)
    axis[1,1].legend(ncol=2)
#    axis[1,1].legend()

    axis[2,0].set_ylabel('Time series')

    #plt.suptitle('Seasonal Mean Jet Latitude')
    plt.tight_layout()
    plt.savefig('plt_season_timeseries.png')
    plt.show()
    pdb.set_trace()



if __name__ == "__main__":
    main()
