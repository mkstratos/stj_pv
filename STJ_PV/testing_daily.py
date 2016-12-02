import numpy as np
import pdb
import os
import collections
from scipy import interpolate
from scipy.signal import argrelmin, argrelmax, argrelextrema
import scipy.io as io
from numpy.polynomial import chebyshev as cby
import copy as copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mstats, t, linregress
from scipy import linalg
import math
from matplotlib.ticker import MultipleLocator
from datetime import date
from general_functions import openNetCDF4_get_data,MeanOverDim, FindClosestElem
from general_plotting import draw_deg

base = os.environ['BASE']
plot_dir = '{}/Plot/Jet'.format(base)

def get_new_lat(hemi):

    if hemi == 'NH':
        new_lat = np.arange(0, 90, 0.5)
    else:
        new_lat = np.arange(-90, 0, 0.5)

    return new_lat

def spline_fit(u,lat,new_lat):

    spline_function = interpolate.interp1d(lat,u, kind='linear')  
    u_spline_fit = spline_function(new_lat)

    return u_spline_fit

def plot_ts(stj_est,edj_est,stj_metric):

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.75])

    time = np.arange(0,stj_est.shape[0],1)
    ax.plot(time,stj_est[:,0],'b', marker = 'x', ls = '', label='NH STJ: max u 250 hPa')
    ax.plot(time,stj_est[:,1],'b', marker = 'x', ls = '', label='SH STJ: max u 250 hPa')
    ax.plot(time,edj_est[:,0],'r', marker = 'x', ls = '', label='NH EDJ: max u 850 hPa')
    ax.plot(time,edj_est[:,1],'r', marker = 'x', ls = '', label='SH EDJ: max u 850 hPa')
    ax.plot(time,stj_metric[:,0],'k', marker = 'x', ls = '-', label='NH STJ metric')
    ax.plot(time,stj_metric[:,1],'k', marker = 'x', ls = '-', label='SH STJ metric')
    lat_array = np.arange(-90, 91, 10)
    lat_plot = draw_deg(lat_array)
    ax.yaxis.set_ticks(lat_array)
    ax.yaxis.set_ticklabels(lat_plot)
    plt.legend(loc=0)
    plt.xticks(np.arange(0, 450, 24),np.arange(1979, 2016, 2))
    plt.yticks(np.arange(-90, 92, 15))
    plt.legend(loc=7, ncol=3, bbox_to_anchor=(0.982, -0.1), numpoints=1)
    plt.savefig('{}/metric_250_850_wind.eps'.format(plot_dir))
    plt.show()
    pdb.set_trace()

def TestDailyData():
    u_fname = '/scratch/pm366/Data/ERA_INT/1979_2015/u79_15.nc'
    var = openNetCDF4_get_data(u_fname)

    lev250 = FindClosestElem(25000, var['lev'],0)[0]
    lev850 = FindClosestElem(85000, var['lev'],0)[0]
    
    zonal_mean_u_250 =  MeanOverDim(data=var['var131'][:,lev250,:], dim=2)  #[time,lat]
    zonal_mean_u_850 =  MeanOverDim(data=var['var131'][:,lev850,:], dim=2)  #[time,lat]

    new_lat_NH = get_new_lat(hemi='NH')
    new_lat_SH = get_new_lat(hemi='SH')

    u_250_spline = np.zeros((zonal_mean_u_250.shape[0],180,2))
    u_250_spline[:,:,0] = spline_fit(u=zonal_mean_u_250,lat=var['lat'],new_lat=new_lat_NH)
    u_250_spline[:,:,1] = spline_fit(u=zonal_mean_u_250,lat=var['lat'],new_lat=new_lat_SH)

    u_850_spline = np.zeros((zonal_mean_u_850.shape[0],180,2))
    u_850_spline[:,:,0] = spline_fit(u=zonal_mean_u_850,lat=var['lat'],new_lat=new_lat_NH)
    u_850_spline[:,:,1] = spline_fit(u=zonal_mean_u_850,lat=var['lat'],new_lat=new_lat_SH)


    stj_est = np.zeros((zonal_mean_u_850.shape[0],2))
    edj_est = np.zeros((zonal_mean_u_850.shape[0],2))
    for i in xrange(2):
      if i == 0:
        new_lat = new_lat_NH
      else:
        new_lat = new_lat_SH

      u250_elem =  np.argmax(u_250_spline[:,:,i], axis=1)   #[time,hemi]
      u850_elem =  np.argmax(u_850_spline[:,:,i], axis=1)
      stj_est[:,i] = new_lat[u250_elem]
      edj_est[:,i] = new_lat[u850_elem]

    file_STJ_metric = '/scratch/pm366/Data/ERA_INT/STJ_data_monthly.nc'
    var_metric = openNetCDF4_get_data(file_STJ_metric)

    stj_metric = var_metric['STJ_lat']
    plot_ts(stj_est,edj_est,stj_metric)

    pdb.set_trace()


if __name__ == "__main__":

    TestDailyData()
