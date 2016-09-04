import numpy as np
import scipy.io as io
import pdb
import os		
from SetDefaults import GetDiri
from cmip5.common.atmos import ertelPV, ipv, theta
from EDJ import Directory
from EDJ_surface_wind import plot_map
from PV_STJ import STJFromPV
from common_modules import openNetCDF4_get_data
from CommonFunctions import MeanOverDim,FindClosestElem,addToList
import collections
import cmip5.common.epv
import matplotlib.pyplot as plt
from plot_routines import get_cmap_for_maps, cbar_Maher
import matplotlib as mpl
from cmip5.common import  staticParams
from scipy import interpolate
import cmip5.common.interp
import platform
import IPV_2max
from instability import OpenPickle, SavePickle

data_name = collections.namedtuple('data_name', 'letter label')
metric = collections.namedtuple('metric', 'name hemisphere intensity position')


class Experiment(object):
  def __init__(self,diri,time_units):
    self.time_units = time_units
    self.diri = diri


  def PrepFilenameERA(self):

    if self.time_units == 'mm':
      time_syn = '_monthly.nc'
    else:
      time_syn = '_daily.nc'
 
    path = self.diri.data_loc + 'Data/ERA_INT/'
    self.u_fname  = path + 'ERA_INT_UWind_correct_levels.nc'
    self.v_fname  = path + 'vwind.nc'
    self.t_fname  = path + 'ERA_int_temp.nc'
    self.path = path

    self.var_names = {'t':data_name(letter='t', label='t'),'u':data_name(letter='u', label='var131'),
      'v':data_name(letter='v', label='var132'), 'p':data_name(letter='p', label='lev')}


    #data window to use: all data
    self.start_time = 0
    self.end_time = 360 
    self.lat_extreme = 90 #interplation blow up if data is not defined. Max to fit spline to

  

def main():

  time_unit_opt = ['dd','mm']
  time_unit = time_unit_opt[1]

  data_options = ['GFDL','Era_CMIP5']
  data_type    = data_options[1]

  RunFlag = ['Open','Save','RunNotSave']
  RunOpt    = RunFlag[0]

  skip_server_warning = True

  #the lat upper limit to what is ignored by the code. Roughly speaking: tropical band from ~-+10
  threshold_lat = 0.0 #-> see also  self.threshold_lat_upper = 70.0

  if skip_server_warning != True:
    assert platform.node() == 'emps-gv1.ex.ac.uk' , 'Data is on gv1 not ' + platform.node() 

  #Identify data files
  diri = Directory()
  Exp = Experiment(diri,time_unit)
  Exp.PrepFilenameERA()

  #filePickle = Exp.path + 'IPV_theta_300_400_every_1_deg.p'   #IPV every 1K between 300-400
  filePickle = Exp.path + 'IPV_data_0_lat.p'                 #IPV every 5K between 300-500
  #filePickle = '/home/pm366/Documents/Data/tmp/IPV_data_0_lat.p'

  if (RunOpt == 'Save') or (RunOpt == 'RunNotSave'):

    Exp.threshold_lat = threshold_lat
    STJ_PV = STJFromPV(Exp)

    #Open the data and calculate IPV
    STJ_PV.OpenFile()
    STJ_PV.GetTropopause()
    STJ_PV.GetIPV()

    if RunOpt == 'Save':
      #output data for faster read in
      STJ_PV.SaveIPV(filePickle)
      
  else: #if open

    IPV_data = OpenPickle(filePickle)

    count = 0 #part of validation
    #Find the STJ metrics
    tmp = IPV_2max.main(IPV_data,count,threshold_lat)
    
 #   STJ_NH,STJ_SH = STJ_PV.Get_uwind_strength()
 
    #Processing results
    STJ_Post_Proc = STJ_Post_Processing()
    STJ_Post_Proc.SaveSTJMetric(STJ_NH,STJ_SH)
    STJ_Post_Proc.PlotIPV(output_plotting)

    filename = Exp.path + 'STJ_metric.nc'
    var  = openNetCDF4_get_data(filename)
    pdb.set_trace()	


  #PV ionstead of IPV - testing only
  #STJ_PV.GetPV()
  #STJ_PV.PlotPV()

  
    
  pdb.set_trace()	
  
  return ()
       
if __name__ == "__main__" : 

  main()
