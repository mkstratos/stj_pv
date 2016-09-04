import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap,addcyclic,shiftgrid
import numpy as np
import pylab 
import math
import pdb
from scipy import ndimage,array
import matplotlib.font_manager as font_manager
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image
import matplotlib.cbook as cbook
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

#personalised
from general_functions import apply_mask_num, addToList,save_file, openfile_get_data


mpl.rc('text', usetex=False)  #turning this flag on/off changes g=hatching with eps
#mpl.rc('font', family='serif') 
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
 


def draw_map_model(data,lon,lat,title,cbar_title,colour,bounds,file_name,show_plot,domain=None, name_cbar=None,coastline=False):
    'Plot a map, latex flags off when using unicode for deg' 

     #Define basic plot parameters
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.1,0.15,0.85,0.85])

    ax.set_title(title)
    #ax.set_title(' %s C' % deg)  #%s means string
    
    m = Basemap(projection='cyl',llcrnrlat=-88,urcrnrlat=86, llcrnrlon=0,urcrnrlon=360,resolution='c')
    m.drawparallels(np.arange(-90.,91.,30.), labels=[0,0,0,0])
    m.drawmeridians(np.arange(0,361.,60.), labels=[0,0,0,0])
    ax.set_xlim(1.5,360)
    ax.set_ylim(-88,86)

    if coastline == True:
      m.drawcoastlines()

    #Note: pcolormesh plots the lower lat and lower lon, not the centre!
    lon,lat=fix_pcolormesh_for_maps(x=lon,y=lat)
    #pdb.set_trace()
    #data, lon = addcyclic(data, lon) 		# so there is no white space at edges lons only

    lon2d, lat2d = np.meshgrid(lon, lat)	# meshed lon lat needed for basemap
    x, y = m(lon2d, lat2d)			# apply mesh
  
    cmap=get_cmap_for_maps(colour=colour,bounds=bounds)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    
    #make the plot
    img=pylab.pcolormesh(x,y,data,cmap=cmap,norm=norm)#, latlon=False)
    #ax_cb=fig.add_axes([0.05, 0.15, 0.92, 0.05])
    #cbar=cbar_Maher(fig,cmap,norm,bounds,cbar_title,ax_cb)
    ax_cb=fig.add_axes([0.1, 0.1, 0.80, 0.05])
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,ticks=bounds, orientation='horizontal')
    cbar.set_label(cbar_title)
    
    if colour != 'BuRd' or colour != 'BuRd_r':
          cmap.set_under('white')
    mpl.rc('text', usetex=False)  #turning this flag on/off changes g=hatching with eps
    fix_ax_label = gfdl_lon_change_map(ax=ax)
    fix_ax_label = gfdl_lat_change_map(ax=ax)   
    

      
    if name_cbar != None:
      cbar.set_ticks(np.array(bounds) + .5)
      cbar.set_ticklabels(name_cbar)


    plt.savefig(file_name)
    
    if show_plot == True:
      plt.show()
 
    print 'Saved plot: ',file_name


    return ()

def plot_map(lon_in,lat_in,colour,bounds,model_type,data,cbar_units,filename,show_plot=None):


    name_cbar = None

    #Define basic plot parameters
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.1,0.15,0.85,0.85])

    ax.set_title('')
    
    m = Basemap(projection='cyl',llcrnrlat=-88,urcrnrlat=86, llcrnrlon=0,urcrnrlon=360,resolution='c')
    m.drawparallels(np.arange(-90.,91.,30.), labels=[0,0,0,0])
    m.drawmeridians(np.arange(0,361.,60.), labels=[0,0,0,0])

    #Note: pcolormesh plots the lower lat and lower lon, not the centre!
    lon,lat=fix_pcolormesh_for_maps(x=lon_in,y=lat_in)
    lon2d, lat2d = np.meshgrid(lon_in, lat_in)	# meshed lon lat needed for basemap
    x, y = m(lon2d, lat2d)			# apply mesh

    cmap=get_cmap_for_maps(colour=colour,bounds=bounds)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax.set_xlim(1.5,360)
    ax.set_ylim(-88,86)

    #make the plot
    img=pylab.pcolormesh(x,y,data,cmap=cmap,norm=norm)#, latlon=False)
    ax_cb=fig.add_axes([0.1, 0.1, 0.80, 0.05])
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,ticks=bounds, orientation='horizontal')
    cbar.set_label(cbar_units)

    if colour != 'BuRd' or colour != 'BuRd_r':
          cmap.set_under('white')
    mpl.rc('text', usetex=False)  #turning this flag on/off changes g=hatching with eps

    fix_ax_label = gfdl_lon_change_map(ax=ax)
    fix_ax_label = gfdl_lat_change_map(ax=ax)   
 
      
    if name_cbar != None:
      cbar.set_ticks(np.array(bounds) + .5)
      cbar.set_ticklabels(name_cbar)

    plt.savefig(filename)
    
    if show_plot == True:
      plt.show()
 
    print 'Saved plot: '


    return fig



def get_cmap_for_maps(colour,bounds):    
    #color for map and colour bar with range
    
    if colour == 'BuRd':
        cmap = mpl.colors.LinearSegmentedColormap.from_list('RdBu_cmap',['Navy','white','Maroon'],N=len(bounds)-1)
    else: 
       #cmap = mpl.colors.ListedColormap(['Yellow','ForestGreen'])
       cmap=pylab.cm.get_cmap(colour) 
       cmap.set_under(color="black")

    return cmap


def fix_pcolormesh_for_maps(x,y):
    'Move grid to plot centre - assumes data has constand dx and dy'
    dx=(x[1]-x[0])*0.5
    dy=(y[1]-y[0])*0.5
    x_new=x+dx
    y_new=y-dy
    return x_new,y_new


def gfdl_lon_change_map(ax):    

    lon_array=np.arange(0,361,60)
    lon_plot=draw_deg(lon_array)
    lon_array[0]=1.41				#avoid white space
    ax.set_xticks(lon_array)
    ax.set_xticklabels(lon_plot)
   
    return()
    
def gfdl_lat_change_map(ax):       
    lat_array=np.arange(-90,91,30)
    lat_plot=draw_deg(lat_array)
    lat_array[0]=-89				
    lat_array[-1]=86				  
    ax.set_yticks(lat_array)
    ax.set_yticklabels(lat_plot)    

    return 

def cbar_Maher(fig,cmap,norm,bounds,cbar_title,ax_cb):
    'Add a horizontal colourbar to a map plot that has a upper and lower triangle rather than ending on a square edge'    
    
    max_tri=(cmap._segmentdata['red'][-1][1],cmap._segmentdata['green'][-1][1],cmap._segmentdata['blue'][-1][1])
    min_tri=(cmap._segmentdata['red'][0][1],cmap._segmentdata['green'][0][1],cmap._segmentdata['blue'][0][1])
    cmap.set_over(max_tri)
    cmap.set_under(min_tri)
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm,extend='both',ticks=bounds,spacing='uniform', orientation='horizontal')
    cbar.set_label(cbar_title)

    return cbar

