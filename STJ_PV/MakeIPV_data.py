import numpy as np
import scipy.io as io
import pdb
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
#Dependent code
import general_functions
from calc_ipv import *
from general_plotting import plot_map, get_cmap_for_maps, cbar_Maher
from thermal_tropopause import TropopauseHeightLevel
from general_functions import openNetCDF4_get_data,MeanOverDim



__author__ = "Penelope Maher"


#created named tuples to manage storage of index
metric = collections.namedtuple('metric', 'name hemisphere intensity position')

base = os.environ['BASE']
plot_dir = '{}/Plot/Jet'.format(base)
if not os.path.exists(plot_dir):
    print('CREATING PLOTTING DIRECTORY: {}'.format(plot_dir))
    os.system('mkdir -p {}'.format(plot_dir))


class Generate_IPV_Data(object):
  def __init__(self,Exp):
    'To avoid calling object within object assign vars of use to this routine'
    self.time_units  = Exp.time_units
    self.diri        = Exp.diri
    self.u_fname     = Exp.u_fname
    self.v_fname     = Exp.v_fname
    self.t_fname     = Exp.t_fname
    self.path        = Exp.path
    self.var_names   = Exp.var_names
    self.start_time  = Exp.start_time
    self.end_time    = Exp.end_time

  def OpenFile(self):
    'Data assumed to be of the format [time, pressure,lat,lon]'

    var_dict= vars(self)

    for fname in var_dict.keys():
      if fname.endswith('fname'):
        #open file and extract data
        var     = openNetCDF4_get_data(var_dict[fname])
        #use the label from named tuple
        label   = self.var_names[fname[0]].label
        label_p = self.var_names['p'].label

        #Ensure surface is the 0'th element
        wh_max_p  = np.where (var[label_p] == var[label_p].max())[0][0]
        flip_pressure = False
        if wh_max_p != 0 :        #if surface is not 0th element then flip data
          flip_pressue = True
          var[label_p] = var[label_p][::-1]
          var[label] = var[label][:,::-1,:,:]

        #add the data to the object
        setattr(self,fname[0],var[label])

    #lon and lat assumed to be on the same grid for each variable
    self.lon = var['lon']
    self.lat = var['lat']

    #check that input pressure is in pascal
    if var[label_p].max()< 90000.0:
      var[label_p] = var[label_p]*100.0
    self.p = var[label_p]

    #restruct data in time from first month to last month. See function PathFilenameERA
    self.t = self.t[self.start_time:self.end_time,:,:,:]
    self.u = self.u[self.start_time:self.end_time,:,:,:]
    self.v = self.v[self.start_time:self.end_time,:,:,:]

    self.time = var['time'][self.start_time:self.end_time]

    print('Finished opening data')


  def GetThermalTropopause(self):
    'Calculate the thermal definition of the tropopause height'

    print('Start calculating tropopause height')

    #Spline fit the temperature in 10hPa intervals from surface to aloft (data must be monotonic increasing)
    #call the external function TropopauseHeightLeve to calculate H

    TropH           = np.zeros([self.end_time,len(self.lat)])
    TropH_pressure  = np.zeros([self.end_time,len(self.lat)])
    TropH_temp      = np.zeros([self.end_time,len(self.lat)])


    t_zonal = MeanOverDim(data=self.t, dim=3)  #[time,level,lat]
    P_spline = np.arange(10,1001,10)           #pressured between 10hPa and 1000 hPa

    for time in range(self.end_time):
      for lat in range(len(self.lat)):
        tck = interpolate.splrep(self.p[::-1]/100,t_zonal[time,:,lat][::-1]) #use hPa instead of Pa
        T_spline = interpolate.splev(P_spline,tck,der=0)

        TropH[time,lat], TropH_pressure[time,lat],TropH_temp[time,lat] = \
           TropopauseHeightLevel(T_spline = T_spline[::-1],  #data from surface to aloft so flip it
                                 P_spline=P_spline[::-1],
                                 tck=tck, T_orig=None,p_orig=None,lat = self.lat )

    self.TropH          = TropH
    self.TropH_pressure = TropH_pressure
    self.TropH_temp     = TropH_temp

    print('Finished calculating tropopause height')

  def GetIPV(self):
    'IPV code interpolates on theta levels'

    print('Starting IPV calculation')

    #calculate IPV.
    IPV,self.p_lev,self.u_th   = ipv(self.u,self.v,self.t,self.p,self.lat,self.lon)

    self.IPV = IPV * 1e6 #units in IPVU

    #310K isopleth often of interest - not currently used
    theta_level_310_K   = np.where(th_levels_trop == 310)[0][0]
    self.p_310          = self.p_lev[:,theta_level_310_K,:,:]
    self.p_310_bar      = MeanOverDim(data=self.p_310[0,:,:], dim=1)

    self.ipv_310     = self.IPV[:,theta_level_310_K,:,:]
    self.ipv_310_bar = MeanOverDim(data=self.ipv_310, dim=0)
    self.theta_lev   = th_levels_trop

    print('Finished calculating IPV')

  def SaveIPV(self,filename_1, filename_2, file_type):
    'Save output as nc or pickle'

    print('Saving ipv data')


    if file_type == '.p':
      output = {}

      output['lat']            = self.lat
      output['lon']            = self.lon
      output['time']           = self.time
      output['theta_lev']      = self.theta_lev

      output['IPV']            = self.IPV
      output['IPV_310']        = self.ipv_310

      output['u']              = self.u_th

      output['TropH']          = self.TropH
      output['TropH_p']        = self.TropH_pressure
      output['TropH_temp']     = self.TropH_temp

      tmp = SavePickle(filenamePickle=filename_1 + file_type, data=output)


    if file_type == '.nc':

      f = io.netcdf.netcdf_file(filename_1+file_type, mode='w')

      #set up dimensions of arrays

      #time
      f.createDimension('time', len(self.time))
      time    = f.createVariable('time','f',('time',))
      time[:] = self.time

      #lat
      f.createDimension('lat', len(self.lat))
      lat    = f.createVariable('lat','f',('lat',))
      lat[:] = self.lat

      #lon
      f.createDimension('lon', len(self.lon))
      lon    = f.createVariable('lon','f',('lon',))
      lon[:] = self.lon

      #theta
      f.createDimension('theta_lev', len(self.theta_lev))
      theta_lev    = f.createVariable('theta_lev','i',('theta_lev',))
      theta_lev[:] = self.theta_lev

      #assign variables

      #IPV
      IPV    = f.createVariable('IPV','f',('time','theta_lev','lat','lon',))
      IPV[:,:,:,:] = self.IPV

      f.close()

      #code was not compatible saving both IPV and u_th so created two files

      f = io.netcdf.netcdf_file(filename_2+file_type, mode='w')

      #set up dimensions of arrays

      #time
      f.createDimension('time', len(self.time))
      time    = f.createVariable('time','f',('time',))
      time[:] = self.time

      #lat
      f.createDimension('lat', len(self.lat))
      lat    = f.createVariable('lat','f',('lat',))
      lat[:] = self.lat

      #lon
      f.createDimension('lon', len(self.lon))
      lon    = f.createVariable('lon','f',('lon',))
      lon[:] = self.lon

      #theta
      f.createDimension('theta_lev', len(self.theta_lev))
      theta_lev    = f.createVariable('theta_lev','i',('theta_lev',))
      theta_lev[:] = self.theta_lev

      #u wind on theta level
      uwnd    = f.createVariable('uwnd','f',('time','theta_lev','lat','lon',))
      uwnd[:,:,:,:] = self.u_th


      #IPV - 310 isopleth
      IPV_310    = f.createVariable('IPV_310','f',('time','lat','lon',))
      IPV_310[:,:,:] = self.ipv_310

      #H
      TropH      = f.createVariable('TropH','f',('time','lat',))
      TropH[:,:]   = self.TropH

      #H p
      TropH_p    = f.createVariable('TropH_p','f',('time','lat',))
      TropH_p[:,:] = self.TropH_pressure

      #H t
      TropH_temp    = f.createVariable('TropH_temp','f',('time','lat',))
      TropH_temp[:,:] = self.TropH_temp

      f.close()

    print('created files: ',filename+file_type, 'and', filename+'_u_H'+file_type)

  def open_ipv_data(self,filename_1, filename_2,file_type):

    if file_type == '.p':
      IPV_data = OpenPickle(filename_1+file_type)
    if file_type == '.nc':
      IPV_data = openNetCDF4_get_data(filename_1+file_type)
      IPV_data_2 = openNetCDF4_get_data(filename_2+file_type)
      IPV_data.update(IPV_data_2)

    self.IPV_data = IPV_data

  def Get_uwind_strength(self):
    'When metric is working then calculate strength'

    time_len = self.IPV.shape[0]
    STJ_int = np.zeros(time_len)
    STJ_pos = np.zeros(time_len)

    for hemi in ['NH','SH']:
      for time_loop in range(time_len):
      #for time_loop in xrange(1):

        if hemi == 'NH':
          lat     = self.lat_NH
          STJ_phi = self.NH_STJ_phi[time_loop]
          STJ_th  = self.NH_STJ_theta[time_loop]
        else:
          lat = self.lat_SH
          STJ_phi = self.SH_STJ_phi[time_loop]
          STJ_th  = self.SH_STJ_theta[time_loop]

        #step 8. interpolate u wind
        u_zonal             = MeanOverDim(data=self.u_th[time_loop,:,:,:],dim=2)
        u_zonal_function    = interpolate.interp2d(self.lat,self.theta_lev, u_zonal, kind='cubic')
        u_zonal_interp      = u_zonal_function(lat,self.theta_interp)

        #step 9: for the 2.0 max derivative latitude find the uwind strength
        #get element closest to phi and theta points

        elem_phi   = FindClosestElem(STJ_phi,lat)[0]
        elem_theta = FindClosestElem(STJ_th,self.theta_interp)[0]

        STJ_int[time_loop] =  u_zonal_interp[elem_theta,elem_phi]
        STJ_pos[time_loop] =  STJ_phi

      if hemi == 'NH':
       Metric_NH = metric(name='STJ', hemisphere=hemi, intensity=STJ_int, position=STJ_pos)
      else:
       Metric_SH = metric(name='STJ', hemisphere=hemi, intensity=STJ_int, position=STJ_pos)

    return Metric_NH,Metric_SH


class STJ_Post_Processing(object):
  def __init__(self):
    pdb.set_trace()


  def SaveSTJMetric(self,STJ_NH,STJ_SH):

    filename = self.path + 'STJ_metric.nc'
    f = io.netcdf.netcdf_file(filename, mode='w')

    #time
    f.createDimension('time', len(STJ_NH.intensity))
    time = f.createVariable('time','i',('time',))
    time[:] = np.arange(0,len(STJ_NH.intensity),1)

    #intensity
    STJI_NH = f.createVariable('STJI_NH','f',('time',))
    STJI_NH[:] = STJ_NH.intensity

    STJI_SH = f.createVariable('STJI_SH','f',('time',))
    STJI_SH[:] = STJ_SH.intensity

    #position
    STJP_NH = f.createVariable('STJP_NH','f',('time',))
    STJP_NH[:] = STJ_NH.position

    STJP_SH = f.createVariable('STJP_SH','f',('time',))
    STJP_SH[:] = STJ_SH.position

    f.close()

    print('created file: ',filename)

    #test file created property
    #var = openNetCDF4_get_data(filename)


  def PlotIPV(self, output_plotting):


    #data prep for plotting
    array_shape_interp = output_plotting['NH','ipv_zonal_interp_0'].shape
    array_shape = output_plotting['NH','ipv_zonal_0'].shape
    lat_array = self.lat[np.newaxis, :] + np.zeros(array_shape)
    theta_lev_array = th_levels_trop[:,np.newaxis] + np.zeros(array_shape)

    #Map of 310 IPV on standard map projection
    filename='{}/IPV_testing_vertical.eps'.format(plot_dir)
    fig1 = plot_map(lon_in=self.lon,lat_in=self.lat,colour='BuRd', bounds = np.arange(-10.,10.5,1.0),
        model_type='Mima',data=self.ipv_310[0,:,:],cbar_units='PVU',filename=filename,show_plot=False)

    #zonal mean IPV with 2PV contour
    filename='{}/IPV_testing_map.eps'.format(plot_dir)
    fig2 = plt.figure(figsize=(8,8))
    ax1 = fig2.add_axes([0.1,0.2,0.75,0.75])
    bounds =  np.arange(-20,22,2.0)
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax1.pcolormesh(lat_array,th_levels_trop,output_plotting['NH','ipv_zonal_0'],cmap=cmap,norm=norm)
    plt.ylim(300,400)
    ax1.set_xlim(-90,90)
    ax_cb=fig2.add_axes([0.1, 0.1, 0.80, 0.05])
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,ticks=bounds, orientation='horizontal')
    cs1 = ax1.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(2,3,1), colors='red' )
    cs2 = ax1.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(-2,-1,1), colors='red' )
    cbar.set_label('PVU')
    ax1.set_title('IPV for 0th month')
    plt.savefig(filename)



    #Contour plots only of IPV
    filename='{}/IPV_contour_test.eps'.format(plot_dir)
    fig3 = plt.figure(figsize=(8,8))
    ax1 = fig3.add_axes([0.1,0.2,0.75,0.75])
    plt.ylim(250,400)
    CS = plt.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(-40,40,1), colors='k' )
    plt.clabel(CS, inline=1,fontsize=8)
    CS = plt.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(2,3,1), colors='red' )
    plt.title('IPV 0th month - 2PV in red')
    plt.savefig(filename)


    #plot zonal mean u on theta levels
    u_plot   = MeanOverDim(data = self.u_th[0,:,:,:], dim=2)
    filename ='{}/IPV_uwind_contour_test.eps'.format(plot_dir)
    fig4     = plt.figure(figsize=(8,8))
    ax1      = fig4.add_axes([0.1,0.2,0.75,0.75])
    cmap     = plt.cm.RdBu_r
    bounds   =  np.arange(-50,51,5.0)
    norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #u wind as a contour
    ax1.pcolormesh(lat_array,theta_lev_array,u_plot,cmap=cmap,norm=norm)
    #plot the 2pv line
    cs1 = ax1.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(2,3,1), colors='red' )
    cs2 = ax1.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(-2,-1,1), colors='red' )
    #plot the fitted data
    plt.plot(output_plotting['NH','phi_2PV'][0,:],output_plotting['NH','theta_2PV'][0,:],marker='x',c='k', linestyle=' ',markersize = 10)
    plt.plot(output_plotting['SH','phi_2PV'][0,:],output_plotting['SH','theta_2PV'][0,:],marker='x',c='k', linestyle=' ',markersize = 10)
    #mark the STJ
    plt.plot(output_plotting['NH','dThdlat_lat'][0],output_plotting['NH','dThdlat_theta'][0],marker='x',c='blue', linestyle=' ',markersize = 16)
    plt.plot(output_plotting['SH','dThdlat_lat'][0],output_plotting['SH','dThdlat_theta'][0],marker='x',c='blue', linestyle=' ',markersize = 16)
    plt.ylabel('Potential Temperature')
    plt.title('U wind. 2PV contour (red). 0th month. X close 2PV (Red max der).')
    plt.ylim(300,400)
    plt.savefig(filename)


    #test domain
    test_data = np.zeros_like(self.theta_domain_array)
    test_data[:,:] = 4.0
    filename ='{}/IPV_uwind_contour_test_domain.eps'.format(plot_dir)
    fig5     = plt.figure(figsize=(8,8))
    ax1      = fig5.add_axes([0.1,0.2,0.75,0.75])
    cmap     = plt.cm.RdBu_r
    bounds   = np.arange(-5,5,1.0)
    norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax1.pcolormesh(self.lat_NH_array, self.theta_domain_array, test_data,cmap=cmap,norm=norm)
    ax1.pcolormesh(self.lat_SH_array, self.theta_domain_array, -test_data,cmap=cmap,norm=norm)
    cs1 = ax1.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(2,3,1), colors='red' )
    cs2 = ax1.contour(lat_array,theta_lev_array,output_plotting['NH','ipv_zonal_0'],levels=np.arange(-2,-1,1), colors='red' )
    plt.plot(output_plotting['NH','phi_2PV'],output_plotting['NH','theta_2PV'],marker='x',c='k', linestyle=' ',markersize = 10)
    plt.plot(output_plotting['NH','dThdlat_lat'][0],output_plotting['NH','dThdlat_theta'],marker='x',c='red', linestyle=' ',markersize = 10)
    plt.ylabel('Potential Temperature')
    plt.title('2PV in red for 0th month in confined domain for testing only')
    plt.ylim(200,450)
    plt.savefig(filename)
    plt.show()


    #check interpolate
    filename ='{}/IPV_test_contour.eps'.format(plot_dir)
    lat_array_interp_NH = self.lat_NH[np.newaxis, :]      + np.zeros(array_shape_interp)
    lat_array_interp_SH = self.lat_SH[np.newaxis, :]      + np.zeros(array_shape_interp)
    theta_array_interp  = self.theta_interp[:,np.newaxis] + np.zeros(array_shape_interp)

    fig6     = plt.figure(figsize=(8,8))
    ax1      = fig6.add_axes([0.1,0.2,0.75,0.75])
    cmap     = plt.cm.RdBu_r
    bounds   =  np.arange(-20,22,2.0)
    norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax1.pcolormesh(lat_array_interp_NH,theta_array_interp,output_plotting['NH','ipv_zonal_interp_0'],cmap=cmap,norm=norm)
    ax1.pcolormesh(lat_array_interp_SH,theta_array_interp,output_plotting['SH','ipv_zonal_interp_0'],cmap=cmap,norm=norm)
    plt.plot(output_plotting['NH','phi_2PV'],output_plotting['NH','theta_2PV'],marker='x',c='k', linestyle=' ',markersize = 10)
    plt.plot(output_plotting['NH','dThdlat_lat'][0],output_plotting['NH','dThdlat_theta'],marker='x',c='red', linestyle=' ',markersize = 10)
    plt.plot(output_plotting['SH','phi_2PV'],output_plotting['SH','theta_2PV'],marker='x',c='k', linestyle=' ',markersize = 10)
    plt.plot(output_plotting['SH','dThdlat_lat'][0],output_plotting['SH','dThdlat_theta'],marker='x',c='red', linestyle=' ',markersize = 10)
    cs1 = ax1.contour(lat_array_interp_NH,theta_array_interp,output_plotting['NH','ipv_zonal_interp_0'],levels=np.arange(2,3,1), colors='red' )
    cs2 = ax1.contour(lat_array_interp_SH,theta_array_interp,output_plotting['SH','ipv_zonal_interp_0'],levels=np.arange(-2,-1,1), colors='red' )
    plt.ylabel('Potential Temperature')
    plt.title('2PV in red for 0th month with interpolated pv')
    plt.ylim(300,400)
    plt.xlim(-90,90)
    ax_cb=fig10.add_axes([0.1, 0.1, 0.80, 0.05])
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,ticks=bounds, orientation='horizontal')
    cbar.set_label('PVU')
    plt.savefig(filename)

    plt.show()
    pdb.set_trace()

    #Get the contour information
    PV2_points = cs1.collections[0].get_paths()[0].vertices
    PV2_pos_lat = PV2_points[:,0]
    PV2_pos_theta = PV2_points[:,1]

    PV2_points = cs2.collections[0].get_paths()[0].vertices
    PV2_neg_lat = PV2_points[:,0]
    PV2_neg_theta = PV2_points[:,1]

    #common elements of list
    #list(set(wh_lat_pos).intersection(wh_theta_250_to_400))



  def ConvertPVUnits(self):
    'PV SI units m2 s-1K kg-1. 1 PVU = 1.0 x 10-6 m2 s-1K kg-1'
    conversion = 1e-6

  def PlotPV(self):

    filename='{}/PVlines.eps'.format(plot_dir)
    bounds = list(range(-25,25,5))
    colour='Blues'
    cbar_units='m/sec'
    Title = 'Testing PV'
    contour_increment = ''

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.1,0.2,0.8,0.75])
    ax.set_title(Title)

    cmap=get_cmap_for_maps(colour=colour,bounds=bounds)
    norm = mpl.colors.Normalize(vmin=np.min(bounds),vmax=np.max(bounds))

    lev = np.arange(np.min(bounds),np.max(bounds),contour_increment)
    CS = plt.contourf(self.lat,self.p,self.PV_zonal,levels = lev, cmap=cmap,norm=norm)
    ax_cb=fig.add_axes([0.05, 0.1, 0.92, 0.05])
    cbar=cbar_Maher(fig,cmap,norm,bounds,cbar_title,ax_cb)

    plt.show()
    pdb.set_trace()
