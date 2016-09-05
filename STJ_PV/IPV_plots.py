import numpy as np
from numpy.polynomial import chebyshev as cby
import scipy.io as io
from scipy.signal import argrelmin, argrelmax, argrelextrema
import time
import matplotlib.ticker as ticker
import pdb
import os		
import matplotlib.pyplot as plt
import matplotlib as mpl
#Dependent code
from general_functions import MeanOverDim, FindClosestElem

__author__ = "Penelope Maher" 

class Plotting(object):

  def __init__(self,data):


    self.dTHdlat_lat     = data.dTHdlat_lat
    self.dtdphi_val      = data.dtdphi_val
    self.phi_2PV         = data.phi_2PV
    self.theta_2PV       = data.theta_2PV
    self.local_elem      = data.local_elem
    self.elem            = data.elem
    #self.peak_sig        = data.peak_sig
    self.lat             = data.lat
    self.theta_lev       = data.theta_lev
    self.TropH_theta     = data.TropH_theta

    #second derivative
    self.local_elem_2    = data.local_elem_2
    self.d2tdphi2_val    = data.d2tdphi2_val
    #STJ lat
    self.STJ_lat         = data.STJ_lat_sort[0]
    self.STJ_lat_all     = data.STJ_lat_sort
    self.cross_lat       = data.cross_lat

    self.lat_NH          = data.lat_NH
    self.lat_SH          = data.lat_SH

    self.shear_elem      = data.shear_elem
    self.shear_max_elem  = data.shear_max_elem

    self.best_guess      = data.best_guess

  def compare_finite_vs_poly(self):

    plt.plot(self.dTHdlat_lat,self.dTHdlat, linestyle='-', c='k',marker='x', markersize=8, label='dTh/dy finite diff')
    plt.plot(self.phi_2PV, self.dtdphi_val, linestyle='-', c='r',marker='.', markersize=8,label='dTh/dy from fit')
    plt.legend()
    plt.ylim(-10,10)
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/cbyfit_vs_finite.eps')
    plt.show()


  def test_second_der(self):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.1,0.2,0.8,0.75])

    ax.plot(self.phi_2PV, self.dtdphi_val, linestyle='-', c='r',marker='.', markersize=8,label='dTh/dy from fit')
    ax.plot(self.phi_2PV, self.d2tdphi2_val, linestyle='-', c='b',marker='.', markersize=8,label='d2Th/dy2 from fit')
    ax.plot(self.phi_2PV[self.local_elem_2],self.d2tdphi2_val[self.local_elem_2] , linestyle=' ',c='b',marker='x', markersize=10,label='d2Th/dy2 peaks')
    ax.plot(self.phi_2PV[self.local_elem],self.dtdphi_val[self.local_elem] , linestyle=' ',c='r',marker='x', markersize=10,label='dTh/dy peaks')
    ax.set_ylim(-5, 5)
    plt.legend(loc=0)

    ax2 = ax.twinx()
    ax2.plot(self.phi_2PV,self.theta_2PV/100., linestyle='-', c='k',marker='x', markersize=8, label='2PV line scaled by x1/100')
    ax2.set_ylim(3, 4)

    plt.legend(loc=0)
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/test_second_der.eps')
    plt.show() 

  def poly_2PV_line(self,hemi,u_zonal,lat_elem,time_loop,pause, click):

    wind_on_plot = True

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_axes([0.07,0.2,0.8,0.75])
    #plot the zonal mean
    if  wind_on_plot == True:
      cmap     = plt.cm.RdBu_r
      bounds   =  np.arange(-50,51,5.0)
      norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
      #u wind as a contour
      ax.pcolormesh(self.lat[lat_elem],self.theta_lev,u_zonal[:,lat_elem][:,0,:],cmap=cmap,norm=norm)
      ax.set_ylabel('Theta')
      ax.set_ylim(300,400)
      ax_cb=fig.add_axes([0.1, 0.1, 0.80, 0.05])
      cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,ticks=bounds, orientation='horizontal')
      cbar.set_label(r'$\bar{u} (ms^{-1})$')

    #ax2 = ax.twinx()
    line6=ax.plot(self.phi_2PV,self.theta_2PV, linestyle='-', c='k',marker='x', markersize=8, label='2PV line - Dynamical Tropopause')
    line7=ax.plot(self.lat, self.TropH_theta[time_loop,:],linestyle='-',c='k',marker='o',label='Tropopause height Lapse rate ')

    #ax2.set_ylabel('2PV line and H Theta ')
    #ax2.set_ylim(300,400)

    ax3 = ax.twinx()
    #ax3.spines['right'].set_position(('axes', 1.07))
    #ax3.set_frame_on(True)
    #ax3.patch.set_visible(False)

    line1=ax3.plot(self.phi_2PV[self.elem], self.dtdphi_val[self.elem], linestyle='-', c='blue',marker='.', markersize=8,label='dTh/dy from fit')
    line2=ax3.plot(self.phi_2PV[self.local_elem],self.dtdphi_val[self.local_elem] , linestyle=' ',c='blue',marker='x', markersize=10,label='dTh/dy peaks')
 #   ax3.plot(self.phi_2PV[self.peak_sig],self.dtdphi_val[self.peak_sig] , linestyle=' ',c='r',marker='o', markersize=10,label='dTh/dy peaks')
    #line3=ax3.plot(self.phi_2PV[self.local_elem_2],self.d2tdphi2_val[self.local_elem_2] , linestyle=' ',c='b',marker='x', markersize=10,label='d2Th/dy2 peaks')
  # line4=ax3.plot(self.phi_2PV,self.d2tdphi2_val , linestyle='-',c='b',label='d2Th/dy2 ')
    line8=ax3.plot([self.cross_lat,self.cross_lat],[-10,10], linestyle=':',c='k',label='Tropo cross')


    ax3.set_ylabel('Derivatived from fit (Blue line)')

    xx = [self.lat[lat_elem][self.shear_elem][self.shear_max_elem],self.lat[lat_elem][self.shear_elem][self.shear_max_elem]]
    yy = [self.theta_lev[0],self.theta_lev[-1]]
    ax.plot(xx,yy, linestyle='-',c='green',linewidth=2,label='dTh/dy peak max shear')

    #lines = line1+line2+line3+line4+line5+line6+line7+line9
    lines = line1+line2+line6+line7+line8

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels,loc=0)

    if hemi == 'NH':
      start, end, inc = 0,91,5
      ax.set_xlim(0, 90)
    else:
      start, end, inc = -0,-91,-5
      ax.set_xlim(-0, -90)
    ax.xaxis.set_ticks(np.arange(start, end, inc))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))

    plt.savefig('/home/links/pm366/Documents/Plot/Jet/looking_at_fit.eps')

    print '  Peaks at: ', self.phi_2PV[self.local_elem], 'ST Jet at :',  self.STJ_lat
  
    if pause == True:
      #pause sequence   
      plt.draw()
      plt.pause(10)
      plt.close()
    else:
      if click == True:
        #button press sequence
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
      else:
        plt.show()
        plt.close()
        pdb.set_trace()



  def timeseries(self):

    pdb.set_trace()
    #plot a timeseries
    print 'Current algorithm plot produces: '
    fig     = plt.figure(figsize=(15,6))
    ax1      = fig.add_axes([0.1,0.2,0.75,0.75]) 
    plt.plot(np.arange(0,STJ_jet_lat[:,0,0].shape[0],1), STJ_jet_lat[:,0,0], c='k',marker='x', markersize=8,linestyle = '-',label='NH')
    plt.plot(np.arange(0,STJ_jet_lat[:,0,0].shape[0],1), STJ_jet_lat[:,0,1] , c='r',marker='x', markersize=8,linestyle = '-',label='SH')
    #plt.legend()
    #plt.ylim(300,380)
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/STJ_ts.eps')
    plt.show()

    pdb.set_trace()



def main():
    
    pdb.set_trace()	
  
    return 
       
if __name__ == "__main__" : 

  main()

  
  if len(self.local_elem) >= 2:

      #check if nearby peak is larger
      if len(self.local_elem) > 2:
             if peak_lat_sort[0] < peak_lat_sort[1]:
               stj_lat_result = self.STJ_lat_sort[1]
               'Second peak is greater - more than 2 peaks ID'
               pdb.set_trace()
             else:
               stj_lat_result = self.STJ_lat_sort[0]


      plot_method_compare = False
      if plot_method_compare == True:
        if self.STJ_lat_sort[0] != self.STJ_lat or len(self.local_elem) >= 2:
             if self.STJ_lat_sort[0] != self.STJ_lat: 
               print 'Methods different'
             if len(self.local_elem) >= 2: 
               print 'More than 2 jets'

          
  else: #only 0 or 1 peaks
    if (print_messages == True) : print 'single jet: ',  self.STJ_lat_sort


    if ((hemi == 'SH') and (self.STJ_lat_sort[0] < -40.0) ) or ((hemi == 'NH') and (self.STJ_lat_sort[0] > 40.0) ) :
      #when STJ is more likely the EDJ
      if hemi == 'SH':
        pdb.set_trace()

     
      save_file_testing  = False
      if save_file_testing == True:
      
        #save data for Mike to test
        np.savez('/home/links/pm366/Documents/Code/Python/Circulation/min_max_example.npz',phi_2PV=self.phi_2PV, dtdphi_val= self.dtdphi_val)
        #test it opens
        npzfile = np.load('/home/links/pm366/Documents/Code/Python/Circulation/min_max_example.npz')
        npzfile.files

def MakeOutputFile(filename,data,dim_name,var_name,var_type):


  f=io.netcdf.netcdf_file(filename, mode='w')
  for j in xrange(dim):
	     f.createDimension(dim_name[j],len(data[dim_name[j]])) 
  for i in xrange(len(var_name)):
        tmp = f.createVariable(var_name[i],var_type[i],var_dim_name[i])
        tmp[:] =  data[var_name[i]]

  f.close()    
  print 'created file: ',filename




