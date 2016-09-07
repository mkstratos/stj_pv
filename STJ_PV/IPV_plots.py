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
from matplotlib.colors import LinearSegmentedColormap  
from colormap import Colormap
#Dependent code
from general_functions import MeanOverDim, FindClosestElem

#see also https://pypi.python.org/pypi/colour
#see https://pypi.python.org/pypi/colormap
__author__ = "Penelope Maher" 

class Plotting(object):

  def __init__(self,data, Method_choice):

    if Method_choice == 'cby':
      #value of fit
      self.pv_fit         = data.theta_cby_val
      self.dxdy            = data.dtdphi_val
      self.dy              = data.phi_2PV

      #local peaks      
      self.local_elem      = data.local_elem_cby

      #elements to poleward side of tropopause crossing
      self.elem            = data.elem_cby

      #second derivative for cby only
      self.local_elem_2    = data.local_elem_2_cby
      self.d2tdphi2_val    = data.d2tdphi2_val

      #STJ lat
      self.STJ_lat         = data.best_guess_cby
      self.STJ_lat_sort    = data.STJ_lat_sort_cby

      self.shear_elem      = data.shear_elem_cby
      self.shear_max_elem  = data.shear_max_elem_cby
      self.jet_max_theta   = data.jet_max_theta_cby

    if Method_choice == 'fd':
      self.dxdy          = data.dTHdlat
      self.dy            = data.dTHdlat_lat

      #local peaks      
      self.local_elem      = data.local_elem_fd

      #elements to poleward side of tropopause crossing
      self.elem            = data.elem_fd

      #second derivative for cby only
      self.local_elem_2    = None
      self.d2tdphi2_val    = None

      #STJ lat
      self.STJ_lat         = data.best_guess_fd
      self.STJ_lat_sort    = data.STJ_lat_sort_fd

      self.shear_elem      = data.shear_elem_fd
      self.shear_max_elem  = data.shear_max_elem_fd
      self.jet_max_theta   = data.jet_max_theta_fd




    #2pv line data
    self.phi_2PV         = data.phi_2PV
    self.theta_2PV       = data.theta_2PV

    self.lat             = data.lat
    self.theta_lev       = data.theta_lev
    self.TropH_theta     = data.TropH_theta
    self.theta_domain    = data.theta_domain
    self.u_fitted        = data.u_fitted

    self.cross_lat       = data.cross_lat
    self.cross_lev       = data.cross_lev

    self.lat_NH          = data.lat_NH
    self.lat_SH          = data.lat_SH
    self.lat_hemi        = data.lat_hemi



  
  def compare_finite_vs_poly(self):

    plt.plot(self.y,self.dTHdlat, linestyle='-', c='k',marker='x', markersize=8, label='dTh/dy finite diff')
    plt.plot(self.phi_2PV, self.dtdphi_val, linestyle='-', c='r',marker='.', markersize=8,label='dTh/dy from fit')
    plt.legend()
    plt.ylim(-10,10)
    plt.savefig('/home/links/pm366/Documents/Plot/Jet/cbyfit_vs_finite.eps')
    plt.show()


  def test_second_der(self):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.1,0.2,0.78,0.75])

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
    ax = fig.add_axes([0.06,0.1,0.85,0.88])
    #plot the zonal mean
    plot_raw_data = False
    if  wind_on_plot == True:
      if plot_raw_data == True:
        cmap     = plt.cm.RdBu_r
        bounds   = np.arange(-50,51,5.0)
        norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #u wind as a contour
        ax.pcolormesh(self.lat[lat_elem],self.theta_lev,u_zonal[:,lat_elem][:,0,:],cmap=cmap,norm=norm)
        ax.set_ylabel('Theta')
        ax.set_ylim(300,400)
        ax_cb=fig.add_axes([0.09, 0.05, 0.60, 0.02])
        cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,ticks=bounds, orientation='horizontal')
        cbar.set_label(r'$\bar{u} (ms^{-1})$')
      else:
        #contour

        cm = Colormap()
        mycmap = cm.cmap_linear('#0033ff', '#FFFFFF', '#990000')  #(neg)/white/(pos)
        levels = np.arange(-60,61,5).tolist()
        ax.contourf(self.lat[lat_elem],self.theta_lev,u_zonal[:,lat_elem][:,0,:], levels, cmap=mycmap)  
        ax.set_ylim(300,400)
        ax_cb=fig.add_axes([0.09, 0.05, 0.80, 0.02])
        norm     = mpl.colors.BoundaryNorm(levels, mycmap.N)
        cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=mycmap,norm=norm,ticks=levels, orientation='horizontal')
        cbar.set_label(r'$\bar{u} (ms^{-1})$',fontsize=16)
        ax.set_ylabel('Theta')


    line6=ax.plot(self.phi_2PV,self.theta_2PV, linestyle='-', c='k',marker='x', markersize=8, label='dynamical tropopause')
    line7=ax.plot(self.lat, self.TropH_theta[time_loop,:],linestyle='-',c='k',marker='.',markersize=4,label='thermodynamic tropopause')
    line8=ax.plot(self.cross_lat,self.cross_lev, linestyle=' ',marker='x',markersize=10,mew=2,c='#009900',label='tropopause crossing')
    line9=ax.plot(self.STJ_lat,self.jet_max_theta, linestyle=' ',marker = 'o',c='#ff3300',markersize=16,markeredgecolor='none',label='Subtropical jet')
    line10=ax.plot(self.phi_2PV, self.pv_fit, linestyle='-', linewidth=1,c='yellow',label='Poly fit')


    ax3 = ax.twinx()
    #move axis off main
    #ax3.spines['right'].set_position(('axes', 1.07))
    #ax3.set_frame_on(True)
    #ax3.patch.set_visible(False)

    line1=ax3.plot(self.dy, self.dxdy, linestyle='-', linewidth=1,c='#0033ff',label=r'$\frac{d \theta}{d \phi}$')
    line2=ax3.plot(self.dy[self.local_elem],self.dxdy[self.local_elem] , linestyle=' ',mew=2,c='#0033ff',marker='x', markersize=12,label=r'peaks')

    ax3.set_ylim(-15,15)

    lines = line1+line2
    labels = [l.get_label() for l in lines]
    loc = (0.65,0.925)
    legend =ax3.legend(lines, labels,loc=loc, fontsize=14, ncol=2,frameon=False,numpoints=1)
    legend.legendHandles[1]._legmarker.set_markersize(8)  #crossing marker size in legend
    #make maths larger
    text = legend.get_texts()[0]
    props = text.get_font_properties().copy()
    text.set_fontproperties(props)
    text.set_size(20)

    ax3.set_ylabel(r'$\frac{d \theta}{d \phi}$',fontsize=26)


    lines = line6+line7+line8+line9+line10

    labels = [l.get_label() for l in lines]
    loc = (0.65,0.8)
    legend=ax.legend(lines, labels,loc=loc, fontsize=14,frameon=False,numpoints=1)
    #set marker size in legend
    legend.legendHandles[2]._legmarker.set_markersize(8)  #crossing marker size in legend
    legend.legendHandles[3]._legmarker.set_markersize(8)  #STJ marker size in legend


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




