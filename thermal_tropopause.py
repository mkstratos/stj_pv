from CommonFunctions import constants
import numpy as np
import pdb


__author__ = "Penelope Maher" 


def IterateCheckTropoHeight(dTdz,dz,T_spline,P_spline,lat):  
  'In the 2km layer above does the lapse rate exceed 2K/km?'
  Threshold = 2.0 #K/km dtdz

  rate_lt_2km = np.where(dTdz <= Threshold)[0]
  tropopause_level = None
  for i in xrange(len(rate_lt_2km)):
    if tropopause_level == None:
      #test each layer satisfying this condition from the bottom up

      #test for a surface inversion
      if P_spline[rate_lt_2km[i]] < 500.0:
        guess_trop_level = rate_lt_2km[i] #maximum pressure
        tropopause_level = LoopTestTropoLevel(dTdz=dTdz,dz=dz,guess_trop_level=guess_trop_level,lat=lat,P_spline=P_spline)

  return tropopause_level

def TropopauseHeightLevel(T_spline,P_spline,tck,T_orig,p_orig,lat):
  'WMO definition of tropopause is the 2K/km threshold.\
  Assumes data is spline fit with monotonically incresing pressure. \
  Assumes data is in hPa.'

 
  con = constants()
  
  dTdz = np.zeros(len(P_spline-1))
  rho = np.zeros(len(P_spline-1))
  rate_change = np.zeros(len(P_spline-1))
  dz = np.zeros(len(P_spline-1))

  Threshold = 2.0 #K/km

  for lev in xrange(len(P_spline)-1): 
    #diff from surface up up
    dT = T_spline[lev+1]-T_spline[lev]         	 # units = K
    dp = (P_spline[lev+1]-P_spline[lev])*100    	# units = Pa
    rho[lev] = (P_spline[lev]*100)/(con['R_d']*T_spline[lev])
    dz[lev] = -dp/(rho[lev]*con['g'])/1000   	# units = km
    dTdz[lev] = (-dT/dz[lev]) 			#K/km
    #print P_spline[lev], dTdz[lev]

  TropHeightIndex = IterateCheckTropoHeight(dTdz=dTdz,dz=dz,T_spline=T_spline,P_spline=P_spline,lat=lat)
  pressure_tropopause = P_spline[TropHeightIndex]
  temperature_tropopause = T_spline[TropHeightIndex]

  test_plot = False
  if test_plot == True:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.15,0.8,0.8]) 
    
    #plot derivatives
    ax.plot(T_spline-273.15,P_spline,c='black',linestyle='-',marker='.',label='')  
    ax.plot(dTdz,P_spline,c='red',linestyle='-',marker='.',label='')
    ax.plot([-100,100],[pressure_tropopause,pressure_tropopause],c='grey',linestyle='--')  
    ax.plot([2,2],[1000,1],c='grey',linestyle='--')      
    update_axis = log_axis(ax,P_spline,labelFontSize="small")
    ax.set_xlim(-80,30)
    ax.set_ylim([1000,10])
    plt.show()
    pdb.set_trace()
  
 
  #Find the derivatives
  #dt  = interpolate.splev(p_orig,tck,der=1)
  #d2t = interpolate.splev(p_orig,tck,der=2)  


  return TropHeightIndex, pressure_tropopause, temperature_tropopause 

