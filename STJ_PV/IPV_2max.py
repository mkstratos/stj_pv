import numpy as np
import pdb
import collections
from scipy import interpolate
from scipy.signal import argrelmin, argrelmax,argrelextrema
from numpy.polynomial import chebyshev as cby
import copy as copy
import matplotlib.pyplot as plt
import matplotlib as mpl
#In addition to libraries
from GetDirectoryPath import GetDiri, Directory
from plotting import draw_map_model
from general_functions import openNetCDF4_get_data,apply_mask_inf,MeanOverDim,FindClosestElem
import calc_ipv  #assigns th_levels_trop
from IPV_plots import Plotting




__author__ = "Penelope Maher" 

#file purpose:  Calculate the subtropical jet strength and position
#using the 2PV contour.

data_name = collections.namedtuple('data_name', 'letter label')
metric = collections.namedtuple('metric', 'name hemisphere intensity position')





class Method_2PV_STJ(object):
  'Input data of the form self.IPV[time,theta,lat,lon]'

  def __init__(self,IPV_data,threshold_lat):


    self.lat             = IPV_data['lat']
    self.lon             = IPV_data['lon']
    self.theta_lev       = IPV_data['theta_lev']
    self.threshold_lat   = threshold_lat
    self.lat_extreme     = IPV_data['lat_extreme']
    self.IPV             = IPV_data['IPV']
    self.TropH_p         = IPV_data['TropH_p']
    self.TropH_temp      = IPV_data['TropH_temp']
    self.TropH_lat       = IPV_data['lat']
    self.u               = IPV_data['u']


  def PrepForAlgorithm(self):

    #interpolate (spline) latitude increment
    lat_increment  = 0.2
    
    #Prep step 1: Define lat and theta to interpolate to
    self.lat_SH = np.arange(-self.lat_extreme,0 + lat_increment, lat_increment)
    self.lat_NH = np.arange(0, self.lat_extreme  + lat_increment, lat_increment)

    # smaller domain to avoid areas where 2.0 PV is not defined
#    self.theta_interp =  np.arange(250,401,1.0)  
    self.theta_domain =  np.arange(310,401,1.0)  

    #Prep step 2: definate the search region in theta for the ST jet 
#    self.wh_theta_domain = np.where((self.theta_interp  <= 400.0) & (self.theta_interp >= 310.0))[0]
#    self.theta_domain    = self.theta_interp[self.wh_theta_domain]

    #mask the array so inf can be managed
    self.IPV = apply_mask_inf(self.IPV)

    #Prep Step 3: isolate the tropics where IPV is inf.
    #lat_elem_NH      = np.where(self.lat > 3.0)[0]
    #lat_elem_SH      = np.where(self.lat < -3.0)[0]
    #lat_elem = (lat_elem_NH).tolist() + (lat_elem_SH).tolist()

    #self.lat = self.lat[lat_elem]
    #self.IPV = self.IPV[:,:,lat_elem,:]


  def Prep_lon_slices(self):


    lon_slices_num = 8.
    if self.lon.min() == 0.0:
      lon_slice = np.arange(0,360,360./lon_slices_num)
    else:
      print 'Check lon data.'
      pdb.set_trace()      
  
    elem_loc = np.zeros(lon_slices_num) 
    for i in xrange(int(lon_slices_num)):
      elem_loc[i] = FindClosestElem(lon_slice[i],self.lon)[0]
  
    self.lon_loc = elem_loc


  def PV_phi_theta(self,time_loop, hemi,slide_method,lon_elem ):

    #Step 1: Take the zonal mean 
    ipv_zonal = MeanOverDim(data=self.IPV[time_loop,:,:,:],dim=2)   #[theta,lat]

    #this is preparing the function for interpolation
    if slide_method == 'zonal_mean':
      ipv_function  = interpolate.interp2d(self.lat, self.theta_lev, ipv_zonal[:,:], kind='cubic')      
    else:
      ipv_function  = interpolate.interp2d(self.lat, self.theta_lev, self.IPV[time_loop,:,:,lon_elem], kind='cubic')


    # - use new lat and theta to interpolate using the function between 310-400K and every 0.2 deg
    #ipv_interp    = ipv_function(self.lat_hemi,self.theta_domain)
    ipv_domain    = ipv_function(self.lat_hemi,self.theta_domain)

    # - reduce the number of plausible theta levels
    #ipv_domain    = ipv_interp[self.wh_theta_domain,:] 

    #Step 4a: Find the element location closest to +-2.0 PV 
    # - code for monotonic increase theta and phi

    line_2PV_elem,actual_ipv_values = IPV_get_2PV(data=ipv_domain, pv_line=self.pv2 ) 

    # - remove zero from array
    line_2PV_list   = line_2PV_elem[line_2PV_elem != 0].tolist()
        
    self.threshold_lat_upper = 90.0
 
    #Next testing
    #print 'Where is the vertical stack condition checked? Also ensure theta above does not hav a theta closer to pole'
    #pdb.set_trace()


    if hemi == 'NH':
          # - include elem that is not near the equator where IPV blows up 
          lat_domain_elem_upper = np.where(self.lat_hemi[line_2PV_list]                        >= self.threshold_lat)[0]

          # - include elem up to 65 but not above as sometimes 2.0 pV line is in polar region
          lat_domain_elem       = np.where(self.lat_hemi[line_2PV_list][lat_domain_elem_upper] < self.threshold_lat_upper)[0]

          #isolate the elements that are near equator  (0-threshold lat) for later use - this is empty if threshold_lat=0
          wh_trivial            = np.where(self.lat_hemi[line_2PV_list]                        < self.threshold_lat)[0]
    else:
          #see comments above
          lat_domain_elem_upper = np.where(self.lat_hemi[line_2PV_list]                        <= -self.threshold_lat)[0]
          lat_domain_elem       = np.where(self.lat_hemi[line_2PV_list][lat_domain_elem_upper] > -self.threshold_lat_upper)[0]
          wh_trivial            = np.where(self.lat_hemi[line_2PV_list]                        > -self.threshold_lat)[0]



    # for latitudes which are between (+-threshold lat and +-threshold upper lat)
    lat_in_domain = self.lat_hemi[line_2PV_list][lat_domain_elem_upper][lat_domain_elem]

    # near the equator identify the abs smallest latitude where thresold is crossed.
    # then not allow any values of lat to be smaller.
    # This avoids pv contours that cross the threshold but are larger aloft of it and are stacked vertically.

    #edge_elem =  np.where ( np.abs(self.lat_hemi[line_2PV_list][ wh_trivial]) == np.abs(self.lat_hemi[line_2PV_list][ wh_trivial]).max())[0]
    #edge_lat = self.lat_hemi[line_2PV_list][ wh_trivial][edge_elem]
   

    #latitudes on poleward side of the threshold lat
    if  len(wh_trivial) != 0:
      lat_domain_elem_no_trivial = np.where(lat_domain_elem < wh_trivial[0])[0]
      elem_satisfied_conditions = lat_domain_elem[lat_domain_elem_no_trivial] 
    else:
      elem_satisfied_conditions = lat_domain_elem 


    #Assign the lat and theta where element is closest to 2.0 PV line (for each theta layer)
    # - when pv starts to get large, closest 2pv is near equator
    phi_2PV    = self.lat_hemi[line_2PV_list][elem_satisfied_conditions]
    theta_2PV  = self.theta_domain[elem_satisfied_conditions] 

    # -  update variables to remove near equator
    line_2PV_list        = line_2PV_elem[elem_satisfied_conditions]       # list of elements of lat spline data that are on 2pv line 
    actual_ipv_values    = actual_ipv_values[elem_satisfied_conditions]   # and the values that are closest to the 2PV lines


    #check the value is to within a certain tolerance - it might not be close to 2.0PV
    #if hemi == 'NH':
    #  cond_upper = actual_ipv_values <= (self.pv2 + self.pv_tolerance)
    #  cond_lower = actual_ipv_values >= (self.pv2 - self.pv_tolerance)
    #  pv2_tol = np.where(np.logical_and(cond_upper,cond_lower) )[0]
    #else:
    #  cond_upper = actual_ipv_values >= (self.pv2 - self.pv_tolerance)
    #  cond_lower = actual_ipv_values <= (self.pv2 + self.pv_tolerance)
    #  pv2_tol = np.where(np.logical_and(cond_upper,cond_lower) )[0]
 
    # - update fields to the set tolerance
    #phi_2PV   = phi_2PV[pv2_tol] 
    #theta_2PV = theta_2PV[pv2_tol]

    #assign to the object the output variables of interest

    self.phi_2PV              = phi_2PV
    self.theta_2PV            = theta_2PV
    self.line_2PV_list        = line_2PV_list 
    self.actual_ipv_values    = actual_ipv_values 
    self.ipv_domain           = ipv_domain
    self.ipv_zonal            = ipv_zonal


  def PolyFit2PV(self):

    
    #remove 0-10 deg for better curve fitting - testing only
    #theta_2PV_in = theta_2PV
    #phi_2PV_in = phi_2PV
    #theta_2PV     = theta_2PV_in[phi_2PV_in > 10]
    #phi_2PV       = phi_2PV_in[phi_2PV_in > 10]


    #find the chebyshev polynomial fit
    theta_cby     = cby.chebfit(self.phi_2PV, self.theta_2PV, 10)
    #values of the fit
    theta_cby_val = cby.chebval(self.phi_2PV, theta_cby)

    #then differentiate d theta_2PV/dy
    dtdphi_cby    = cby.chebder(theta_cby)
    #values of the derivative d theta d phi
    dtdphi_val    = cby.chebval(self.phi_2PV, dtdphi_cby)

    #second derivative
    d2tdphi2_cby    = cby.chebder(dtdphi_cby)
    d2tdphi2_val    = cby.chebval(self.phi_2PV, d2tdphi2_cby)






    self.theta_cby_val = theta_cby_val
    self.dtdphi_val    = dtdphi_val 
    self.d2tdphi2_val  = d2tdphi2_val

  def Poly_testing(self):

      #Plot the fit
      plt.plot(self.phi_2PV, self.theta_2PV, c='k',marker='x', markersize=8,linestyle = '-',label='Data')
      plt.plot(self.phi_2PV, self.theta_cby_val, c='r',marker='.', markersize=8,linestyle = '-',label='cby')
      plt.legend()
      plt.ylim(300,380)
      plt.savefig('/home/links/pm366/Documents/Plot/Jet/cbyfit_10.eps')
      plt.show()

      #plot the derivative to identify local maxima in.
      plt.plot(self.phi_2PV, self.dtdphi_val,   label='dTh/dy')
      plt.plot(self.phi_2PV, self.d2tdphi2_val, label='d2Th/dy2')
      plt.legend()
      plt.ylim(-10,10)
      plt.savefig('/home/links/pm366/Documents/Plot/Jet/cbyfit_10_derivative.eps')
      plt.show()

      pdb.set_trace()


  def unique_elements(self):
    #this fit can return repeated pairs of theta and phi. 

    #Remove repeated elements.
    phi_2PV, phi_idx = np.unique(self.phi_2PV, return_index=True)

    #sort the elements to ensure the array order is the same then apply to all arrays 
    phi_idx            = np.sort(phi_idx)
    self.phi_2PV       = self.phi_2PV[phi_idx]
    self.theta_2PV     = self.theta_2PV[phi_idx]
    self.dtdphi_val    = self.dtdphi_val[phi_idx]
    self.d2tdphi2_val  = self.d2tdphi2_val[phi_idx]
    self.theta_cby_val = self.theta_cby_val[phi_idx]

    #test if there are two d_theta values that are the same next to each other and if so remove one.
    theta_2PV_test1, idx_test1,idx_test2 = np.unique(self.theta_2PV, return_inverse=True,return_index  =True)
    if len(self.theta_2PV) != len(theta_2PV_test1):
      'Investigate this condition' 
      pdb.set_trace()

    #dtdphi_val, idx = np.unique(dtdphi_val_in, return_index=True)
    #idx = np.sort(idx)
    #dtdphi_val = dtdphi_val_in[phi_idx]


  def PolyFit2PV_peaks(self,print_messages,hemi,time_loop):

    # restrict data from equ to +20 to avoid the derivative from getting large.self.phi_2PV[self.local_elem]
    # ID number of peaks in SH and trough in NH

    #using the polynomial first derivative
    x = self.phi_2PV
    y = self.dtdphi_val
    #using the polynomial second derivative
    x2 = self.phi_2PV
    y2 = self.d2tdphi2_val

    theta_max = self.TropH_theta[time_loop,:]
 

    cross_lat = TropoCrossing(hemi,self.TropH_theta[time_loop,:],self.TropH_lat,self.theta_2PV,self.phi_2PV,time_loop)

    #do not keep first two or last two points as they often have large changes in the fit

    elem, local_elem, local_elem_2,   x_peak, y_peak , x2_peak, y2_peak  = IsolatePeaks(hemi,x,y,y2,theta_max,time_loop,cross_lat,print_messages=True)


    #using the finite dif method
    a1 = self.dTHdlat_lat
    b1 = self.dTHdlat

    elem_finite, local_elem_finite, tmp,  x_peak_finite, y_peak_finite, tmp , tmp  = IsolatePeaks(hemi,a1,b1,None,theta_max,time_loop,cross_lat,print_messages=False)
   

    if hemi == 'SH':
         print 'peak out of loop is ',self.phi_2PV[local_elem]

    #assignment from polynomial fit
    # - data from 10deg up
    #self.theta_2PV                          = self.theta_2PV[elem]  
    #self.phi_2PV                            = self.phi_2PV[elem]  
    #self.dtdphi_val                         = self.dtdphi_val[elem]  
    #self.d2tdphi2_val                       = self.d2tdphi2_val[elem]  

    self.elem,       self.elem_finite       = elem, elem_finite
    self.local_elem, self.local_elem_finite = local_elem, local_elem_finite 
    self.local_elem_2                       = local_elem_2

    self.phi_2PV_peak,    self.dtdphi_val_peak      = x_peak, y_peak 
    self.phi_2PV_peak_dt2,self.dtdphi_val_peak_dt2  = x2_peak, y2_peak
    #self.local_elem_first_peak = local_elem_first_peak
    self.cross_lat = cross_lat

    #assignment from finite difference 
    self.elem_finite, self.local_elem_finite = elem_finite, local_elem_finite 
    self.x_peak_finite, self.y_peak_finite   = x_peak_finite, y_peak_finite

  def PolyFit2PV_SortedLat(self,hemi,print_messages):

    if hemi == 'NH':
      sort_index = np.argsort(self.phi_2PV_peak)
      STJ_lat_sort = np.sort(self.phi_2PV_peak) 
      #use unsorted array     
      peak_mag = (self.dtdphi_val_peak).min()

      #sort_index_sig = np.argsort(self.peak_sig)
      #STJ_lat_sort_sig = np.sort(self.peak_sig) 
    else:
      sort_index = np.argsort(self.phi_2PV_peak)[::-1]
      STJ_lat_sort = np.sort(self.phi_2PV_peak)[::-1]
      #use unsorted array     
      peak_mag = (self.dtdphi_val_peak).max()

     # sort_index_sig = np.argsort(self.peak_sig)[::-1]
    #  STJ_lat_sort_sig = np.sort(self.peak_sig)[::-1]


    if (print_messages == True) : print 'Method two:'            
    if (print_messages == True) : print 'STJ: ', STJ_lat_sort[0], ',    EDJ: ',STJ_lat_sort[1], ',    Other: ', STJ_lat_sort[2:]

    self.sort_index       = sort_index 
    self.STJ_lat_sort     = STJ_lat_sort
    self.peak_mag         = peak_mag 
   # self.peak_sig_sort    = sort_index_sig
   # self.STJ_lat_sort_sig = STJ_lat_sort_sig


 

  def PolyFit2PV_near_mean(self,print_messages,STJ_mean,EDJ_mean):
    'Find the peaks that are closest to known mean position. Tests if ordered or ID peaks different.'

    local_lat = self.phi_2PV_peak.tolist()
    local_elem_cp =  copy.deepcopy(self.local_elem)

    #find the peaks closest to know location
    STJ_elem =  FindClosestElem(STJ_mean,np.array(local_lat))
    STJ_lat_near_mean  =  local_lat[STJ_elem] 

    if len(local_lat) >= 2:
      #don't identify the same element so remove it from the sample
      local_elem_cp.remove(local_elem_cp[STJ_elem])  
      local_lat.remove(local_lat[STJ_elem])

      EDJ_elem = FindClosestElem(EDJ_mean,np.array(local_lat))
      EDJ_lat_near_mean =  local_lat[EDJ_elem]

      local_elem_cp.remove(local_elem_cp[EDJ_elem])
      local_lat.remove(local_lat[EDJ_elem])

      Additional_lat  = local_lat

    else:
      EDJ_lat_near_mean , Additional_lat = None,None

    if (print_messages == True) : print 'Method one:'            
    if (print_messages == True) : print 'STJ: ', STJ_lat_near_mean , ',    EDJ: ', EDJ_lat_near_mean, ',    Other: ', Additional_lat

    self.STJ_lat_near_mean = STJ_lat_near_mean
    self.EDJ_lat_near_mean = EDJ_lat_near_mean 
    self.Additional_lat = Additional_lat

  def SeasonalPeaks(self,seasons, STJ_array,crossing_lat):

    STJ_seasons   = np.zeros([STJ_array.shape[0]/4,STJ_array.shape[1],4])
    cross_seasons = np.zeros([crossing_lat.shape[0]/4,crossing_lat.shape[1],4])
    #this is an overkil way to treat the seasons but has reduced risk of mixing up seasons
    count_DJF,count_MAM,count_JJA,count_SON = 0,0,0,0
    for i in xrange(STJ_array.shape[0]):
      if seasons[i] == 'DJF':
         STJ_seasons[count_DJF,:,0]   = STJ_array[i,:]
         cross_seasons[count_DJF,:,0] = crossing_lat[i,:]
         count_DJF = count_DJF + 1
      if seasons[i] == 'MAM':
         STJ_seasons[count_MAM,:,1]   = STJ_array[i,:]
         cross_seasons[count_MAM,:,1] = crossing_lat[i,:]
         count_MAM = count_MAM + 1
      if seasons[i] == 'JJA':
         STJ_seasons[count_JJA,:,2]   = STJ_array[i,:]
         cross_seasons[count_JJA,:,2] = crossing_lat[i,:]
         count_JJA = count_JJA + 1
      if seasons[i] == 'SON':
         STJ_seasons[count_SON,:,3]   = STJ_array[i,:]
         cross_seasons[count_SON,:,3] = crossing_lat[i,:]
         count_SON = count_SON + 1
#      if i%3 == 0 and i !=0:
#        count = count + 1

    output = {}
    output['DJF'] = STJ_seasons[:,:,0] #month,hemi,season
    output['MAM'] = STJ_seasons[:,:,1]
    output['JJA'] = STJ_seasons[:,:,2]
    output['SON'] = STJ_seasons[:,:,3]

    self.STJ_seasons = output    

    cross = {}
    cross['DJF'] = cross_seasons[:,:,0] #month,hemi,season
    cross['MAM'] = cross_seasons[:,:,1]
    cross['JJA'] = cross_seasons[:,:,2]
    cross['SON'] = cross_seasons[:,:,3]

    self.STJ_seasons = output    
    self.cross_seasons = cross    


    print '--------------------------------Assess data---------------------------------------------'
    print ' DJF NH: ', output['DJF'][:,0].min(), output['DJF'][:,0].max(), output['DJF'][:,0].mean()
    print ' DJF SH: ', output['DJF'][:,1].min(), output['DJF'][:,1].max(), output['DJF'][:,1].mean()
    print ' MAM NH: ', output['MAM'][:,0].min(), output['MAM'][:,0].max(), output['MAM'][:,0].mean()
    print ' MAM SH: ', output['MAM'][:,1].min(), output['MAM'][:,1].max(), output['MAM'][:,1].mean()
    print ' JJA NH: ', output['JJA'][:,0].min(), output['JJA'][:,0].max(), output['JJA'][:,0].mean()
    print ' JJA SH: ', output['JJA'][:,1].min(), output['JJA'][:,1].max(), output['JJA'][:,1].mean()
    print ' SON NH: ', output['SON'][:,0].min(), output['SON'][:,0].max(), output['SON'][:,0].mean()
    print ' SON SH: ', output['SON'][:,1].min(), output['SON'][:,1].max(), output['SON'][:,1].mean()



    fig = plt.figure(figsize=(14,8))
    ax = fig.add_axes([0.1,0.2,0.8,0.75])
    ax.plot(output['DJF'][:,0],'blue',   label='NH Winter'+('  {0:.2f}').format(output['DJF'][:,0].mean()), ls = ' ', marker='x')
    ax.plot(output['JJA'][:,1],'blue',   label='SH Winter'+(' {0:.2f}').format(output['JJA'][:,1].mean()), ls = ' ', marker='x')
    ax.plot(output['MAM'][:,0],'green',  label='NH Spring'+('  {0:.2f}').format(output['MAM'][:,0].mean()), ls = ' ', marker='x')
    ax.plot(output['SON'][:,1],'green',  label='SH Spring'+(' {0:.2f}').format(output['SON'][:,1].mean()), ls = ' ', marker='x')
    ax.plot(output['SON'][:,0],'orange', label='NH Autumn'+('  {0:.2f}').format(output['SON'][:,0].mean()), ls = ' ', marker='x')
    ax.plot(output['MAM'][:,1],'orange', label='SH Autumn'+(' {0:.2f}').format(output['MAM'][:,1].mean()), ls = ' ', marker='x')
    ax.plot(output['JJA'][:,0],'red',    label='NH Summer'+('  {0:.2f}').format(output['JJA'][:,0].mean()), ls = ' ', marker='x')
    ax.plot(output['DJF'][:,1],'red',    label='SH Summer'+(' {0:.2f}').format(output['DJF'][:,1].mean()), ls = ' ', marker='x')
    plt.legend(loc=7,ncol=4,bbox_to_anchor=(1.0, -0.1))
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/index_ts.eps')
    plt.show()


    #plot the crossing points

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_axes([0.1,0.2,0.8,0.75])
    ax.plot(cross['DJF'][:,0],'blue',   label='NH Winter', ls = ' ', marker='x')
    ax.plot(cross['JJA'][:,1],'blue',   label='SH Winter', ls = ' ', marker='x')
    ax.plot(cross['MAM'][:,0],'green',  label='NH Spring', ls = ' ', marker='x')
    ax.plot(cross['SON'][:,1],'green',  label='SH Spring', ls = ' ', marker='x')
    ax.plot(cross['SON'][:,0],'orange', label='NH Autumn', ls = ' ', marker='x')
    ax.plot(cross['MAM'][:,1],'orange', label='SH Autumn', ls = ' ', marker='x')
    ax.plot(cross['JJA'][:,0],'red',    label='NH Summer', ls = ' ', marker='x')
    ax.plot(cross['DJF'][:,1],'red',    label='SH Summer', ls = ' ', marker='x')
    ax.set_ylim(-25, 25)
    plt.legend(loc=7,ncol=4,bbox_to_anchor=(1.0, -0.1))
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/cross.png')
    plt.show()

    pdb.set_trace()

  def validate_near_mean(self,hemi_count,season,input_string,hemi, time_loop):

    #test each month to see if the STJ lat is anywhere near the mean.
    #if it is not near it then print it.
    
    mean_val = self.STJ_seasons[season][:,hemi_count].mean()

    if np.abs(self.STJ_lat_near_mean)-np.abs(mean_val) >= np.abs(5.0): 
      print 'Seasonal mean is ', mean_val, ' but current jet lat is ', self.STJ_lat_near_mean, 'for ', input_string
      PlottingObject = Plotting(Method) 
      best_guess_jet = PlottingObject.poly_2PV_line(hemi,time_loop, pause = False)

      pdb.set_trace()

    #the distance between the jets should be 15 deg. But which peaks are kept and thrown away. 
    #keep_peak = np.zeros(len(Method.phi_2PV_peak)-1)
    #for i in xrange(len(Method.phi_2PV_peak)-1):
    #  if Method.phi_2PV_peak[i+1] - Method.phi_2PV_peak[i] > np.abs(15.0):
    #    pdb.set_trace()
         #keep_peak[i] =  


  def TropopauseTheta(self):

    #Find the tropopause height
    #theta_level_310_K   = np.where(get_ipv.th_levels_trop == 310)[0][0]
    #ipv_310 = MeanOverDim(data=self.IPV[time_loop,theta_level_310_K,:,:], dim=1)

    Rd = 287.0
    Cp = 1004.0
    K = Rd / Cp  
   
    self.TropH_theta = self.TropH_temp* (1000.0/self.TropH_p) ** K
    


  def AttemptMeanFind(self,hemi,time_loop):
    #Isolate data between first and last peak
    detrended = scipy.signal.detrend(self.dtdphi_val[self.elem],type='linear')

    max_peaks = argrelmax(detrended)[0].tolist()
    min_peaks = argrelmin(detrended)[0].tolist()

    peaks     = np.sort(max_peaks+min_peaks)

    peak_min_val,peak_max_val = detrended[peaks].min(), detrended[peaks].max()

    below_max = np.where(detrended <=peak_max_val)[0]
    above_below_max = np.where(detrended[below_max] >= peak_min_val)[0]
    
    valid_data = detrended[below_max][above_below_max]
    signal_normalised =  valid_data.mean() /np.std(valid_data)

    plt.plot(self.phi_2PV[self.elem],self.dtdphi_val[self.elem], c='k',linestyle = '-',label='Data')
    plt.plot(self.phi_2PV[self.elem],detrended, c='k',marker='x', markersize=8,linestyle = '-',label='Data detrended')
    plt.plot(self.phi_2PV[above_below_max],detrended[above_below_max], c='r',marker='x', markersize=8,linestyle = '-',label='Data detrended restricted')
    plt.plot([20,50],[valid_data.mean() ,valid_data.mean() ], c='r',linestyle = '--',label='Mean')
    plt.legend()
    plt.show() 

    pdb.set_trace()
    #Normalized to unit variance by removing the long-term mean and dividing by the standard deviation.
    valid_range_remove_mean = valid_range-valid_range.mean()
    valid_range_normal = valid_range_remove_mean/np.std(valid_range_remove_mean) #sigma of this is 1.0

    plt.plot(self.phi_2PV[peaks.min():peaks.max()],valid_range_normal, c='k',marker='x', markersize=8,linestyle = '-',label='Data detrended')


 
    signal_above_mean = np.where(np.abs(dtdphi_val_normal) >= 0.2)[0].tolist()
    #keep peaks that are in above list
    peak_sig = []
    for i in xrange(len(local_elem)):
      if local_elem[i] in signal_above_mean:
        peak_sig.append(local_elem[i])

    pdb.set_trace()
    #Plot the fit
    plt.plot(self.phi_2PV,data, c='k',marker='x', markersize=8,linestyle = '-',label='Data scaled z1/100')
    plt.plot(self.phi_2PV[peak_sig],dtdphi_val_normal[peak_sig], c='r',marker='.', markersize=8,linestyle = ' ',label='peaks')
    plt.plot(self.phi_2PV[local_elem],dtdphi_val_normal[local_elem], c='r',marker='x', markersize=8,linestyle = ' ',label=' allpeaks')
    #plt.legend()
    plt.show() 
  

  def PolyFit2PV_testing(self, PlottingObject,hemi,u_zonal,time_loop):


    if self.STJ_lat_sort[0] != self.STJ_lat_near_mean:
      print 'STJ_lat_sort[0] != STJ_lat', self.STJ_lat_sort[0], self.STJ_lat_near_mean

    peaks         = self.dtdphi_val_peak
    peak_elem     = np.where(peaks == self.peak_mag)[0]
    peak_lat      = self.phi_2PV.tolist()[peak_elem] 
    peak_lat_sort = peaks[self.sort_index]

    #plot the poly fit and the 2PV line
    
    best_guess_jet = PlottingObject.poly_2PV_line(hemi,u_zonal,time_loop,pause = True)

    return best_guess_jet

  def PlotTesting(self,time_loop):


    diri = Directory()
    path = diri.data_loc + 'Data/ERA_INT/'
    u_fname  = path + 'ERA_INT_UWind_correct_levels.nc'
    #u_fname  = '/home/pm366/Documents/Data/tmp/ERA_INT_UWind_correct_levels.nc'

    var  = openNetCDF4_get_data(u_fname)

    uwind168_surf = var['var131'][time_loop,-1,:,:]
    uwind168_top  = var['var131'][time_loop,11,:,:]
    uwind168_del  = var['var131'][time_loop,11,:,:] -var['var131'][time_loop,-1,:,:] 


    filename_surf ='/home/links/pm366/Documents/Plot/Jet/uwind_'+str(time_loop)+'_1000.eps'
    filename_top ='/home/links/pm366/Documents/Plot/Jet/uwind_'+str(time_loop)+'_250.eps'
    filename_del ='/home/links/pm366/Documents/Plot/Jet/uwind_'+str(time_loop)+'_del.eps'


    plot = draw_map_model(uwind168_surf,var['lon'],var['lat'],'Surface (1000)','cbar','RdBu_r', np.arange(-30,30,5.0),filename_surf,False,coastline=True)
    plot = draw_map_model(uwind168_top, var['lon'],var['lat'],'250','cbar','RdBu_r', np.arange(-60,60,5.0),filename_top,False,coastline=True)
    plot = draw_map_model(uwind168_del, var['lon'],var['lat'],'250-1000','cbar','RdBu_r', np.arange(-60,60,5.0),filename_del,False,coastline=True)


    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.1,0.2,0.8,0.75])

    ax.plot(self.phi_2PV, self.dtdphi_val, linestyle='-', c='r',marker='.', markersize=8,label='dTh/dy from fit')
    ax.plot(self.phi_2PV, self.d2tdphi2_val, linestyle='-', c='b',marker='.', markersize=8,label='d2Th/dy2 from fit')
    ax.plot(self.phi_2PV_peak_dt2 ,self.dtdphi_val_peak_dt2 , linestyle=' ',c='b',marker='x', markersize=10,label='d2Th/dy2 peaks')
    ax.plot(self.phi_2PV_peak ,self.dtdphi_val_peak , linestyle=' ',c='r',marker='x', markersize=10,label='dTh/dy peaks')
    ax.set_ylim(-5, 5)
    plt.legend(loc=0)

    ax2 = ax.twinx()
    ax2.plot(self.phi_2PV,self.theta_2PV/100., linestyle='-', c='k',marker='x', markersize=8, label='2PV line scaled by x1/100')
    ax2.set_ylim(3, 4)

    plt.legend(loc=0)
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/test_second_der'+str(time_loop)+'.eps')
    plt.show()


  def PV_differentiate(self):
    'Finite difference method'
 

    #Step 6: Differentiate to find steepest local maximum of a near-2PV elements

    non_zero_len = len(self.theta_2PV)
    dTHdlat      = np.zeros(non_zero_len-1)
    dTHdlat_lat  = np.zeros(non_zero_len-1)  #latitude for phi between points

    for PV_line_loop in xrange(non_zero_len-1):  
       dTH                      = self.theta_2PV[PV_line_loop+1] - self.theta_2PV[PV_line_loop] 
       dlat                     = self.phi_2PV[PV_line_loop+1]   - self.phi_2PV[PV_line_loop] 
       if dlat == 0.0: 
         #when multiple theta have the same lat, add a small increment so that derivative is not inf.
         dlat  = 0.01
       dTHdlat[PV_line_loop]    = np.abs(dTH/dlat)
       dTHdlat_lat[PV_line_loop] = (self.phi_2PV[PV_line_loop+1] - self.phi_2PV[PV_line_loop])/2. + self.phi_2PV[PV_line_loop]


    plot_finite_diff = False
    if plot_finite_diff == True:    
      plt.plot(self.phi_2PV,self.theta_2PV)
      plt.savefig('/home/links/pm366/Documents/Plot/Jet/testing1.eps')
      plt.show()

      plt.plot(dTHdlat_lat,dTHdlat)
      plt.savefig('/home/links/pm366/Documents/Plot/Jet/finite_diff_test.eps')
      plt.show()
      pdb.set_trace()
 
    self.dTHdlat_lat  = dTHdlat_lat 
    self.dTHdlat      = dTHdlat 


#  def PV_phi_theta_max(self,time_loop,hemi,phi_2PV,theta_2PV,dTHdlat):


    # - first estimate is that the steepest element is STJ but need to identify if this works.
#    slope_max_elem = np.where(dTHdlat == dTHdlat.max())[0][0]

    #how many turning points are there?

#    pdb.set_trace()

#    z = np.polyfit(phi_2PV, theta_2PV, 4)
#    f = np.poly1d(z)
    #calculate new x's and y's
#    x_new = np.linspace(phi_2PV[0], phi_2PV[-1], 100) 
#    y_new = f(x_new)

#    plt.plot(phi_2PV, theta_2PV, 'o', x_new, y_new)
#    plt.xlim([phi_2PV[0]-1, phi_2PV[-1] + 1 ])
#    plt.show()




    #identify local maxima and minima
    #if hemi == 'NH' and time_loop == 247:
    #       local_maxima = argrelextrema(dTHdlat, np.greater)
    #       local_minima = argrelextrema(dTHdlat, np.less)
    #       pdb.set_trace()
     
    #Step 7. at the max derivative identify the lat and theta

#    theta_2PV_max = theta_2PV[slope_max_elem]
#    phi_2PV_max   = phi_2PV[slope_max_elem]

#    pdb.set_trace()

#    return theta_2PV_max, phi_2PV_max



  def test_method_plot(self,time_loop,u_th,ipv_interp,lat,ipv_zonal,phi_2PV,theta_2PV,dThdlat_lat,dThdlat_theta,phi_2PV_max,theta_2PV_max):
    'Investigate if STJ lat is identified in tropics or extratropics'
    u_plot   = MeanOverDim(data = u_th[time_loop,:,:,:], dim=2)
    array_shape = u_plot.shape
    lat_array = self.lat[np.newaxis, :] + np.zeros(array_shape)
    theta_lev_array = get_ipv.th_levels_trop[:,np.newaxis] + np.zeros(array_shape)

    array_shape_interp = ipv_interp.shape
    lat_array_interp = lat[np.newaxis, :]      + np.zeros(array_shape_interp)
    theta_array_interp  = self.theta_interp[:,np.newaxis] + np.zeros(array_shape_interp)

    filename ='/home/links/pm366/Documents/Plot/Jet/IPV_uwind_contour_test_cases.eps'
    fig     = plt.figure(figsize=(8,8))
    ax1      = fig.add_axes([0.1,0.2,0.75,0.75]) 
    cmap     = plt.cm.RdBu_r
    bounds   =  np.arange(-50,51,5.0)
    norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax1.pcolormesh(lat_array,theta_lev_array,u_plot,cmap=cmap,norm=norm)
    #ax1.pcolormesh(lat_array_interp,theta_array_interp,ipv_zonal_interp,cmap=cmap,norm=norm)
    cs1 = ax1.contour(lat_array,theta_lev_array,ipv_zonal,levels=np.arange(2,3,1), colors='red' )
    cs2 = ax1.contour(lat_array,theta_lev_array,ipv_zonal,levels=np.arange(-2,-1,1), colors='red' )
    plt.plot(phi_2PV, theta_2PV, marker='x',c='k', linestyle=' ',markersize = 10)
    plt.plot(dThdlat_lat[time_loop], dThdlat_theta[time_loop],  marker='x',c='red', linestyle=' ',markersize = 10)
    #plt.plot(dThdlat_lat[local_maxima], dThdlat_theta[local_maxima],  marker='x',c='blue', linestyle=' ',markersize = 10)
    #plt.plot(dThdlat_lat[local_minima], dThdlat_theta[local_minima],  marker='x',c='green', linestyle=' ',markersize = 10)

    plt.xticks(np.arange(-90.0, 91.0, 15))
    plt.ylabel('Potential Temperature')
    plt.title('U wind. 2PV contour (red). 0th month. X close 2PV (Red max der).')
    plt.ylim(300,400)
    plt.savefig(filename)
    plt.show()
    pdb.set_trace()

def IPV_get_2PV(data, pv_line):

      array_len         = data.shape[0]
      line_2PV_elem     = np.zeros(array_len)
      actual_ipv_values = np.zeros(array_len)

      for i in xrange(array_len):
        data_loc                      = data[i,:]
        elem                          = FindClosestElem(pv_line,data_loc)[0]
        line_2PV_elem[i]              = elem
        actual_ipv_values[i]          = data[i,elem]
  

      return line_2PV_elem, actual_ipv_values
def func(x, a, b, c):
     return a * np.exp(-b * x) + c

def TropoCrossing(hemi, H,H_lat, theta_IPV,phi_IPV,time_loop):
  #find the intersection of 2.0PV line and thermal tropopause

  #plt.plot(H_lat_hemi, H_hemi, '.', phi_IPV,theta_IPV , 'x')
  #plt.show()

 # if time_loop == 82:
 #   pdb.set_trace()

  if hemi == 'NH':
    elem = np.where(H_lat > 0.0)[0]
  else:
    elem = np.where(H_lat < 0.0)[0]

  H_hemi     = H[elem]
  H_lat_hemi = H_lat[elem]

  if hemi == 'NH':
    min_range = max([phi_IPV.min(),H_lat_hemi.min()])
    max_range = min([phi_IPV.max(),H_lat_hemi.max()])
    new_lat = np.arange(max_range,min_range,-1)
  else:
    min_range = max([phi_IPV.min(),H_lat_hemi.min()])
    max_range = min([phi_IPV.max(),H_lat_hemi.max()])
    new_lat = np.arange(min_range,max_range,1)

  #interpolate tropopause height and 2 IPV line to every 1 deg
  spline_function  = interpolate.interp1d(H_lat_hemi, H_hemi, kind='linear')  #the spline fit did not work well    
  h_spline_fit = spline_function(new_lat)
  spline_function2  = interpolate.interp1d(phi_IPV, theta_IPV, kind='linear') 
  IPV_spline_fit = spline_function2(new_lat)

#  plt.plot(H_lat_hemi, H_hemi, '.', new_lat, h_spline_fit , 'x')
#  plt.plot(phi_IPV,theta_IPV, '+',new_lat,IPV_spline_fit, 'o')
#  plt.show()
 
  diff = np.abs(h_spline_fit - IPV_spline_fit)
  loc = np.where(diff == diff.min())[0]
  cross_lat = new_lat[loc]


  if np.abs(cross_lat) > 30:
    fig1 = plt.figure(figsize=(14,8))
    ax1 = fig1.add_axes([0.1,0.15,0.8,0.8]) 
    ax1.plot(H_lat_hemi,H_hemi, c='blue',linestyle=' ',marker='.',label='Thermal tropopause')    
    ax1.plot(new_lat,h_spline_fit, c='blue',linestyle=' ',marker='x',label='Thermal tropopause linear fit') 
    ax1.plot(phi_IPV,theta_IPV, c='red',linestyle=' ',marker='.',label='Dynamic tropopause') 
    ax1.plot(new_lat,IPV_spline_fit, c='red',linestyle=' ',marker='x',label='Dynamic tropopause linear fit') 
    ax1.plot([cross_lat,cross_lat],[100,500], c='green',linestyle='-',label='Crossing Lat') 
    plt.show()
    plt.close()
    pdb.set_trace()


  return cross_lat


def IsolatePeaks(hemi,x,y,y2,theta_max,time_loop,cross_lat,print_messages): 

    #data assumes to be 90-0 N and -90-0S
    #only keep data to the poleward side of first min or max. Also remove last element of data

    if hemi == 'NH':
  
        elem_lower     = np.where((x >= cross_lat))[0]
        #elem_range     = np.where((x[elem_lower] <= 70.))[0]
        #local_elem_first_peak  = (argrelmax(y[elem_range])[0]).tolist()[-1] #this is the smallest lat -> x[elem_range][local_elem_first_peak]
        #elem                   = elem_range[1:local_elem_first_peak] #allowable range for jet is from peak closest to equator to pole
        elem                   = np.where((x[elem_lower] <= 70.))[0]
        local_elem             = elem[(argrelmin(y[elem])[0])].tolist()
 
        if y2 != None:
          local_elem_2     = (argrelmin(y2[elem])[0]).tolist()  

        #if (print_messages == True) : print  hemi ,':  (local min)' 

    else:

        elem_lower          = np.where((x  <= cross_lat) )[0]
        #elem_range          = np.where((x[elem_lower] >= -70.))[0]
        #local_elem_first_peak  = (argrelmin(y[elem_range])[0]).tolist()[-1]
        #elem                   = elem_range[1:local_elem_first_peak] #allowable range for jet is from peak closest to equator to pole
        elem                   = np.where((x[elem_lower] >= -70.))[0]
        local_elem             = elem[(argrelmax(y[elem])[0])].tolist()

        if y2 != None:
          local_elem_2      = (argrelmax(y2[elem])[0]).tolist()
       # if (print_messages == True) : print hemi ,':  (local max)'

    #x, y               = x[elem], y[elem]               
    x_peak, y_peak     = x[local_elem], y[local_elem] 

    if y2 != None:  
      #y2 = y2[elem]
      x2_peak, y2_peak = x[local_elem_2], y2[local_elem_2]     

    else:
      x2_peak, y2_peak, local_elem_2 = None, None, None

    if (print_messages): #and time_loop == 11):
      print '-----------------------------'
      print hemi, ' peak is', x_peak
      print hemi, ' elements', x[elem]
      print '-----------------------------'

      if len(x_peak) == 0:
        pdb.set_trace()

    return  elem, local_elem, local_elem_2, x_peak, y_peak , x2_peak , y2_peak#,local_elem_first_peak 

def season_mask(time_len):

  mask   = np.zeros(time_len)
  mask_number = np.zeros(time_len)
  time_tmp = np.arange(0,time_len,1)
  mask = []
  for i in  time_tmp: 
    year = i/int(12)
    month = i-year*12
    if month == 0 or month == 1 or month == 11: 
      mask.append('DJF')  
      mask_number[i] = 0 
    if month >=2 and month <= 4: 
      mask.append('MAM')  
      mask_number[i] = 1 
    if month >=5 and month <= 7: 
      mask.append('JJA')  
      mask_number[i] = 2 
    if month >=8 and month <= 10:
      mask.append('SON') 
      mask_number[i] = 3 


  return  mask,mask_number

def MakeOutfileSavez_grid(filename,lat,theta,u,h):

  #save data for Mike to test
  np.savez(filename,lat=lat,theta=theta,u_zonal=u,H_thermal=h)
  print 'File created: ', filename
  #test it opens
  npzfile = np.load(filename)
  npzfile.files   
  
  pdb.set_trace()

def MakeOutfileSavez_derived(filename, phi_2PV,theta_2PV,dth,dth_lat,d2th):


  #save data for Mike to test
  np.savez(filename,lat=phi_2PV, theta=theta_2PV,dth=dth,dth_lat=dth_lat,d2th=d2th)
  print 'File created: ', filename
  #test it opens
  npzfile = np.load(filename)
  npzfile.files   
  
  pdb.set_trace()


def main(IPV_data,count,threshold_lat):
    'Input assumed to be a dictionary'
 
    output_plotting = {}

    Method = Method_2PV_STJ(IPV_data,threshold_lat)
    Method.PrepForAlgorithm()  
    Method.TropopauseTheta()

    slide_method_opt = ['zonal_mean', 'lon_slices'] #for now only the zonal mean should be used
    slide_method = slide_method_opt[0]

    if slide_method == 'lon_slices':
      #For each month, take longitude slices to find the max slope in 2.0 IPV
      Method.Prep_lon_slices()  
    else:
      Method.lon_loc = [0.0] #just to make loops work once (needed for compatability with lon slice method) 
      lon_elem = None

    #Hemispherically separate to produce jet metrics for each hemisphere
    phi_2PV_out       = np.zeros([Method.IPV.shape[0],2,Method.theta_domain.shape[0]])  #[time, hemi, theta is restriced domain]
    theta_2PV_out     = np.zeros([Method.IPV.shape[0],2,Method.theta_domain.shape[0]])

    dth_out            = np.zeros([Method.IPV.shape[0],2,Method.theta_domain.shape[0]])
    d2th_out           = np.zeros([Method.IPV.shape[0],2,Method.theta_domain.shape[0]])
    dth_lat_out        = np.zeros([Method.IPV.shape[0],2,Method.theta_domain.shape[0]])
    #d2th_lat_out       = np.zeros([Method.IPV.shape[0],2,Method.theta_domain.shape[0]])

    
    dThdlat_lat       = np.zeros([Method.IPV.shape[0],2])
    dThdlat_theta     = np.zeros([Method.IPV.shape[0],2])
    STJ_jet_lat       = np.zeros([Method.IPV.shape[0],len(Method.lon_loc),2])
    jet_best_guess    = np.zeros([Method.IPV.shape[0],len(Method.lon_loc),2])
    mask_jet_number   = np.zeros([Method.IPV.shape[0],2])
    STJ_array         = np.zeros([Method.IPV.shape[0],2])   
    crossing_lat      = np.zeros([Method.IPV.shape[0],2])


    #fill with nans
    phi_2PV_out[:,:,:]    = np.nan
    theta_2PV_out[:,:,:]  = np.nan 
    dth_out[:,:,:]        = np.nan
    dth_lat_out[:,:]      = np.nan 
    d2th_out[:,:,:]       = np.nan 
    dth_lat_out[:,:]      = np.nan 
    #d2th_lat_out[:,:]     = np.nan 

    #assign the seasons to array 
    seasons,seasons_num = season_mask(Method.IPV.shape[0])
    
    #save u and h for mike
    save_file = False
    if save_file == True:  
      u_zonal_all_time = MeanOverDim(data=Method.u[:,:,:,:],dim=3)  
      filename = '/home/links/pm366/Documents/Data/STJ_PV_metric.npz'
      MakeOutfileSavez_grid(filename,Method.lat,Method.theta_lev,u_zonal_all_time,Method.TropH_theta)

    print_messages = False
    count_num_single_jets = 0
 
    for time_loop in xrange(Method.IPV.shape[0]):
      for lon_loop in xrange(len(Method.lon_loc)):
        for hemi in ['NH','SH']:
          if hemi == 'NH':
            Method.lat_hemi = Method.lat_NH
            Method.pv2 = 2.0
            hemi_count = 0
            STJ_mean, EDJ_mean = 30.,50.
          else:
            Method.lat_hemi = Method.lat_SH
            Method.pv2 = -2.0
            hemi_count = 1
            STJ_mean, EDJ_mean = -30.,-50.

          #Method.pv_tolerance = 0.02
   
          if slide_method == 'lon_slices': lon_elem = Method.lon_loc[lon_loop]

          Method.PV_phi_theta(time_loop,hemi,slide_method,lon_elem)

          #finite difference derivative
          Method.PV_differentiate()

          #polynomial fit and derivative twice
          Method.PolyFit2PV()

          #test the poly fit
          #Method.Poly_testing()

          #Method can have repeated theta, phi pairs. Remove if so
          Method.unique_elements()

          #Method.
          #phi_2PV,theta_2PV,ipv_interp,ipv_zonal, dTHdlat,dTHdlat_lat,theta_cby_val, dtdphi_val,d2tdphi2_val,phi_2PV_output,theta_2PV_output

          #for the poly fitted data find the peaks in the 2 pv line and remove any elements equatorside of 20 deg
          #theta_2PV_fit, elem, local_elem, local_elem_dt2, phi_2PV_fit, dtdphi_val, d2tdphi2_val, phi_2PV_peak, dtdphi_val_peak, phi_2PV_peak_d2t =
          Method.PolyFit2PV_peaks(print_messages, hemi,time_loop)

          #test if peaks are different to mean
          #Method.AttemptMeanFind(hemi,time_loop)


          #sort the peaks
          Method.PolyFit2PV_SortedLat(hemi,print_messages) #phi_2PV_peak,dtdphi_val_peak

          #find the peak closest to estimated mean - used for testing
          Method.PolyFit2PV_near_mean(print_messages,STJ_mean,EDJ_mean)

          STJ_lat = Method.STJ_lat_sort[0] 

          #how many jets were ID for each month and hemisphere
          mask_jet_number[time_loop,hemi_count] = len(Method.local_elem)
          STJ_array[time_loop, hemi_count] = STJ_lat
          crossing_lat[time_loop, hemi_count] = Method.cross_lat

          phi_2PV_out  [time_loop,hemi_count,0:len(Method.phi_2PV)]   =  Method.phi_2PV
          theta_2PV_out[time_loop,hemi_count,0:len(Method.phi_2PV)]   =  Method.theta_2PV 
          dth_out      [time_loop,hemi_count,0:len(Method.elem)]      =  Method.dtdphi_val[Method.elem] 
          dth_lat_out  [time_loop,hemi_count,0:len(Method.elem)]      =  Method.phi_2PV[Method.elem]

          d2th_out     [time_loop,hemi_count,0:len(Method.d2tdphi2_val[Method.elem])]  =  Method.d2tdphi2_val[Method.elem]
          #d2th_lat_out [time_loop,hemi_count,0:len(Method.d2tdphi2_val)]  =  Method.d2tdphi2_val - same as dth_lat_out
 
          #get the zonal wind 
          u_zonal             = MeanOverDim(data=Method.u[time_loop,:,:,:],dim=2)   

          #test with plotting
          if time_loop >=0 :
            test_with_plots = True
          else:
            test_with_plots = False

          if test_with_plots == True:
            print 'plot for: hemi', hemi, ', time: ', time_loop
            PlottingObject = Plotting(Method) 
            #test if peak closest to equator is near known STJ mean
            jet_best_guess[time_loop,lon_loop,hemi_count] = Method.PolyFit2PV_testing(PlottingObject,hemi,u_zonal, time_loop)

    if save_file == True:  
      filename = '/home/links/pm366/Documents/Data/STJ_PV_metric_derived.npz'
      MakeOutfileSavez_derived(filename, phi_2PV_out,theta_2PV_out,dth_out,dth_lat_out,d2th_out)


    #seasonally seperate the data
    Method.SeasonalPeaks(seasons, jet_best_guess[:,0,:],crossing_lat)

    pdb.set_trace()

    #second pass - loop at peaks that are not near the seasonal mean
    for time_loop in xrange(Method.IPV.shape[0]):
      for lon_loop in xrange(len(Method.lon_loc)):
        for hemi in ['NH','SH']:
          if hemi == 'NH':
            hemi_count = 0
          else:
            hemi_count = 1
          input_string = str(hemi)+str(':  ')+str(time_loop)
          Method.validate_near_mean(hemi_count,seasons[time_loop],input_string,hemi,time_loop)

    print count
    pdb.set_trace() 


          #theta_2PV_max, phi_2PV_max =  Method.PV_phi_theta_max(time_loop, lon_loop,hemi,phi_2PV,theta_2PV)  #now code that differentiates

          #at the max derivative identify the lat and theta

          #dThdlat_lat  [time_loop,lon_loop,hemi_count]  = phi_2PV_max 
          #dThdlat_theta[time_loop,lon_loop,hemi_count]  = theta_2PV_max
          

          #if np.abs(dThdlat_lat[time_loop]) < 20 or np.abs(dThdlat_lat[time_loop]) > 45:
          #if hemi == 'NH' and time_loop == 247:
          #if 1 == 1:
          #Method.test_method_plot(time_loop,STJ_PV.u_th,ipv_interp,Method.lat_hemi ,ipv_zonal,phi_2PV,theta_2PV,dThdlat_lat,dThdlat_theta,phi_2PV_max,theta_2PV_max)

        #if time_loop == 0:
        #  output_plotting[hemi,'ipv_0']        = ipv_zonal
        #  output_plotting[hemi,'ipv_interp_0'] = ipv_interp
 
      #output_plotting[hemi,'dThdlat_lat']        = dThdlat_lat
      #output_plotting[hemi,'dThdlat_theta']      = dThdlat_theta
      #output_plotting[hemi,'phi_2PV']            = phi_2PV_out
      #output_plotting[hemi,'theta_2PV']          = theta_2PV_out


    #keep hemisphereic seperated data for lat and theta
#    self.NH_STJ_phi      = output_plotting['NH','dThdlat_lat']
#    self.SH_STJ_phi      = output_plotting['SH','dThdlat_lat']
#    self.NH_STJ_theta    = output_plotting['NH','dThdlat_theta']
#    self.SH_STJ_theta    = output_plotting['SH','dThdlat_theta']


    #for plot testing only
#    self.lat_NH_array       = self.lat_NH[np.newaxis, :]  + np.zeros(ipv_zonal_domain.shape)
#    self.lat_SH_array       = self.lat_SH[np.newaxis, :]  + np.zeros(ipv_zonal_domain.shape)
#    self.theta_domain_array = self.theta_domain[:,np.newaxis] + np.zeros(ipv_zonal_domain.shape)
    
               #use the polynomial fitted data peaks to find the steapest part of 2PV line and hence a jet core.
          #Keep in mind that a merged jet state is still technically the latitude of the STJ but the interpretation is just different (i.e. is the STJ defined in summer)
          #keep in mind the goal is not to ID EDJ and STJ it is just the STJ
    
    pdb.set_trace()	
  
    return output_plotting
       
if __name__ == "__main__" : 

  main()


