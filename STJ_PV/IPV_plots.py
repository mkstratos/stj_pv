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
from colormap import Colormap
from matplotlib import rc
import copy as copy
import math
import os.path
# Dependent code
from general_functions import (MeanOverDim, FindClosestElem, openNetCDF4_get_data,
                               latex_table)
from general_plotting import draw_map_model, draw_deg, gfdl_lat_change_map
# see also https://pypi.python.org/pypi/colour
# see https://pypi.python.org/pypi/colormap


rc('text', usetex=True)

__author__ = "Penelope Maher"

base = os.environ['BASE']
plot_dir = '{}/Plot/Jet'.format(base)
data_out_dir = '{}/Data'.format(base)
u_data_dir = '/scratch/pm366/Data/ERA_INT/1979_2015/'

if not os.path.exists(plot_dir):
    print('CREATING PLOTTING DIRECTORY: {}'.format(plot_dir))
    os.system('mkdir -p {}'.format(plot_dir))

if not os.path.exists(data_out_dir):
    print('CREATING DATA OUT DIRECTORY: {}'.format(data_out_dir))
    os.system('mkdir -p {}'.format(data_out_dir))


class Plotting(object):

    def __init__(self, data, Method_choice):

        if Method_choice == 'cby':
            # value of fit
            self.pv_fit = data.theta_cby_val
            self.dxdy = data.dtdphi_val
            self.dy = data.phi_2PV

            # local peaks
            self.local_elem = data.local_elem_cby

            # elements to poleward side of tropopause crossing
            self.elem = data.elem_cby

            # second derivative for cby only
            self.local_elem_2 = data.local_elem_2_cby
            self.d2tdphi2_val = data.d2tdphi2_val

            # STJ lat
            self.STJ_lat = data.best_guess_cby
            self.STJ_lat_sort = data.STJ_lat_sort_cby

            self.shear_elem = data.shear_elem_cby
            self.shear_max_elem = data.shear_max_elem_cby
            self.jet_max_theta = data.jet_max_theta_cby

        if Method_choice == 'fd':
            self.dxdy = data.dTHdlat
            self.dy = data.dTHdlat_lat

            # local peaks
            self.local_elem = data.local_elem_fd

            # elements to poleward side of tropopause crossing
            self.elem = data.elem_fd

            # second derivative for cby only
            self.local_elem_2 = None
            self.d2tdphi2_val = None

            # STJ lat
            self.STJ_lat = data.best_guess_fd
            self.STJ_lat_sort = data.STJ_lat_sort_fd

            self.shear_elem = data.shear_elem_fd
            self.shear_max_elem = data.shear_max_elem_fd
            self.jet_max_theta = data.jet_max_theta_fd

        # 2pv line data
        self.phi_2PV = data.phi_2PV
        self.theta_2PV = data.theta_2PV

        self.lat = data.lat
        self.theta_lev = data.theta_lev
        self.TropH_theta = data.TropH_theta
        self.theta_domain = data.theta_domain
        self.u_fitted = data.u_fitted

        self.cross_lat = data.cross_lat
        self.cross_lev = data.cross_lev

        self.lat_NH = data.lat_NH
        self.lat_SH = data.lat_SH
        self.lat_hemi = data.lat_hemi

        # self.AnnualCC        = data.AnnualCC
        # self.AnnualPC        = data.AnnualPC
        # self.MonthlyCC       = data.MonthlyCC
        # self.MonthlyPC       = data.MonthlyPC
        # self.CalendarCC      = data.CalendarCC
        # self.CalendarPC      = data.CalendarPC


    def compare_finite_vs_poly(self):

        plt.plot(self.y, self.dTHdlat, linestyle='-', c='k',
                 marker='x', markersize=8, label='dTh/dy finite diff')
        plt.plot(self.phi_2PV, self.dtdphi_val, linestyle='-', c='r',
                 marker='.', markersize=8, label='dTh/dy from fit')
        plt.legend()
        plt.ylim(-10, 10)
        plt.savefig('{}/cbyfit_vs_finite.eps'.format(plot_dir))
        plt.show()

    def test_second_der(self):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.1, 0.2, 0.78, 0.75])

        ax.plot(self.phi_2PV, self.dtdphi_val, linestyle='-', c='r',
                marker='.', markersize=8, label='dTh/dy from fit')
        ax.plot(self.phi_2PV, self.d2tdphi2_val, linestyle='-', c='b',
                marker='.', markersize=8, label='d2Th/dy2 from fit')
        ax.plot(self.phi_2PV[self.local_elem_2], self.d2tdphi2_val[self.local_elem_2],
                linestyle=' ', c='b', marker='x', markersize=10, label='d2Th/dy2 peaks')
        ax.plot(self.phi_2PV[self.local_elem], self.dtdphi_val[self.local_elem],
                linestyle=' ', c='r', marker='x', markersize=10, label='dTh/dy peaks')
        ax.set_ylim(-5, 5)
        plt.legend(loc=0)

        ax2 = ax.twinx()
        ax2.plot(self.phi_2PV, self.theta_2PV / 100., linestyle='-', c='k',
                 marker='x', markersize=8, label='2PV line scaled by x1/100')
        ax2.set_ylim(3, 4)

        plt.legend(loc=0)
        plt.savefig('{}/test_second_der.eps'.format(plot_dir))
        plt.show()

    def poly_2PV_line(self, hemi, u_zonal, lat_elem, time_loop, fig, ax, plot_cbar, ax_cb,
                      plot_type, pause, click, save_plot):

        #set specifics depending on if making a single plot or subplots
        if plot_type == 'single':
            legend_font = 16
            label_font  = 18
            loc1 = (0.65, 0.925)
            loc2 = (0.65, 0.78)
            jet_marker_size   = 16
            cross_marker_size = 14
            peak_marker_size  = 12
        else:
            legend_font = 12
            label_font  = 18
            loc1 = (0.42, 0.92)
            loc2 = (0.42, 0.76)
            jet_marker_size   = 10
            cross_marker_size = 10
            peak_marker_size  = 8
        wind_on_plot = True

        # plot the zonal mean
        plot_raw_data = False
        if wind_on_plot:
            if plot_raw_data:
                cmap = plt.cm.RdBu_r
                bounds = np.arange(-50, 51, 5.0)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                # u wind as a contour
                ax.pcolormesh(self.lat[lat_elem],
                              self.theta_lev, u_zonal[:, lat_elem][:, 0, :],
                              cmap=cmap, norm=norm)
                ax.set_ylabel('Theta')
                ax.set_ylim(300, 400)
                if plot_cbar == True:
                  cbar = mpl.colorbar.ColorbarBase(
                      ax_cb, cmap=cmap, norm=norm, ticks=bounds, orientation='horizontal')
                  cbar.set_label(r'$\bar{u} (ms^{-1})$')
            else:
                # contour
                cm = Colormap()
                # (neg)/white/(pos)
                mycmap = cm.cmap_linear('#0033ff', '#FFFFFF', '#990000')
                levels = np.arange(-60, 61, 5).tolist()
                ax.contourf(self.lat[lat_elem], self.theta_lev,
                            u_zonal[:, lat_elem][:, 0, :], levels, cmap=mycmap)
                ax.set_ylim(300, 400)

                norm = mpl.colors.BoundaryNorm(levels, mycmap.N)

                #ax.xaxis.set_visible(False)
                #remove labels and ticks
                if plot_cbar == True:
                  cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=mycmap, norm=norm,
                                                 ticks=levels, orientation='horizontal')
                  cbar.set_label(r'$\bar{u} (ms^{-1})$', fontsize=16)


        line6 = ax.plot(self.phi_2PV, self.theta_2PV, linestyle='-', c='k',
                        marker='x', markersize=8, label='dynamical H')
        line7 = ax.plot(self.lat, self.TropH_theta[time_loop, :], linestyle='-', c='k',
                        marker='.', markersize=4, label='thermodynamic H')
        line8 = ax.plot(self.cross_lat, self.cross_lev, linestyle=' ', marker='o',
                        mfc='none', c='#006600',
                        markersize=cross_marker_size, mew=2, label='tropopause crossing')
        line9 = ax.plot(self.STJ_lat, self.jet_max_theta, linestyle=' ', marker='o',
                        c='#0033ff', markersize=jet_marker_size, markeredgecolor='none',
                        label='Subtropical jet')
        line10 = ax.plot(self.phi_2PV[self.elem], self.pv_fit[self.elem], linestyle=':',
                         linewidth=1, c='k', label='Poly fit')

        ax3 = ax.twinx()
        # move axis off main
        # ax3.spines['right'].set_position(('axes', 1.07))
        # ax3.set_frame_on(True)
        # ax3.patch.set_visible(False)

        line1 = ax3.plot(self.dy[self.elem], self.dxdy[self.elem], linestyle='-', linewidth=1,
                         c='#0033ff', label=r'$\frac{d \theta}{d \phi}$')
        line2 = ax3.plot(self.dy[self.local_elem], self.dxdy[self.local_elem],
                         linestyle=' ', mew=2, c='#0033ff', marker='x', markersize=peak_marker_size,
                         label=r'peaks')

        ax3.set_ylim(-15, 15)

        if (plot_type == 'single') or (hemi == 'NH'):

            #first legend: derivative and peaks
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            legend = ax3.legend(lines, labels, loc=loc1, fontsize=legend_font,
                            ncol=2, frameon=False, numpoints=1)
            legend.legendHandles[1]._legmarker.set_markersize(
                8)  # crossing marker size in legend

            # make maths larger
            text = legend.get_texts()[0]
            props = text.get_font_properties().copy()
            text.set_fontproperties(props)
            text.set_size(18)

            #All other lines
            lines2 = line6 + line7 + line8 + line9 + line10
            labels2 = [l.get_label() for l in lines2]
            legend2 = ax.legend(lines2, labels2, loc=loc2,fontsize=legend_font,
                               frameon=False, numpoints=1,ncol=1)
            # set marker size in legend
            legend2.legendHandles[2]._legmarker.set_markersize(
                8)  # crossing marker size in legend
            legend2.legendHandles[3]._legmarker.set_markersize(8)  # STJ marker size in legend



        if (plot_type == 'single') or (hemi == 'SH'):
            ax.set_ylabel(r'$\theta$',rotation=0, fontsize=label_font )
            #turn off other label axis
            ax3.set_yticklabels('')
        if (plot_type == 'single') or (hemi == 'NH'):
            ax3.set_ylabel(r'$\frac{d \theta}{d \phi}$', fontsize=label_font+4,rotation=0)
            #turn off other label axis
            ax.set_yticklabels('')
                  

        if hemi == 'NH':
            if plot_type == 'single':
                start, end, inc = 0, 91, 5
                ax.set_xlim(0, 90)
            if plot_type == 'subplot':
                start, end, inc = 0, 91, 10
                ax.set_xlim(0, 90)

        else:
            if plot_type == 'single':
                start, end, inc = 0, -91, -10
                ax.set_xlim(0, -90)
            if plot_type == 'subplot':
                start, end, inc = -90, 1, 10
                ax.set_xlim(-90,0)

        ax.xaxis.set_ticks(np.arange(start, end, inc))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))

        print('  Peaks at: ', self.phi_2PV[self.local_elem], 'ST Jet at :', self.STJ_lat)

        if save_plot == True:
            plt.savefig('{}/looking_at_fit.eps'.format(plot_dir))
        if pause or click :
            show_plot(pause, click)



    def timeseries(self):

        pdb.set_trace()
        # plot a timeseries
        print('Current algorithm plot produces: ')
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_axes([0.1, 0.2, 0.75, 0.75])
        plt.plot(np.arange(0, STJ_jet_lat[:, 0, 0].shape[0], 1), STJ_jet_lat[:, 0, 0],
                 c='k', marker='x', markersize=8, linestyle='-', label='NH')
        plt.plot(np.arange(0, STJ_jet_lat[:, 0, 0].shape[0], 1), STJ_jet_lat[:, 0, 1],
                 c='r', marker='x', markersize=8, linestyle='-', label='SH')
        # plt.legend()
        # plt.ylim(300,380)
        plt.savefig('{}/STJ_ts.eps'.format(plot_dir))
        plt.show()

def plot_validation_for_paper(Method, u_zonal, method_choice, 
                              plot_subplot, hemi, time_loop, lat_elem,
                              Method_NH, u_zonal_NH,lat_elem_NH):


    PlottingObject = Plotting(Method, method_choice)
    if plot_subplot == False:

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_axes([0.06, 0.1, 0.85, 0.88])
        ax_cb = fig.add_axes([0.09, 0.05, 0.60, 0.02])
        plot_cbar = True
        PlottingObject.poly_2PV_line(hemi, u_zonal, lat_elem, time_loop, fig, ax, plot_cbar, ax_cb, 
                                     plot_type = 'single', pause=False, click=True, save_plot=True)
    else:

        print 'Plotting subplot'

        fig = plt.figure(figsize=(10, 10))
        #SH
        ax1 = plt.subplot2grid((20, 21), (0, 0), colspan=10, rowspan=18)
        #NH
        ax2 = plt.subplot2grid((20, 21), (0, 11), colspan=10, rowspan=18)
        #cb
        ax_cb = plt.subplot2grid((20, 21), (19, 0), colspan=21)

        plot_cbar = True 
        # SH and colorbar
        PlottingObject.poly_2PV_line('SH', u_zonal, lat_elem, time_loop, fig, ax1, plot_cbar, ax_cb,
                                           plot_type = 'subplot',pause=False, click=False, save_plot=False)

        # NH
        PlottingObject2 = Plotting(Method_NH, method_choice)
        plot_cbar = False
        PlottingObject2.poly_2PV_line('NH', u_zonal_NH, lat_elem_NH, time_loop, fig, ax2, plot_cbar, None, 
                                            plot_type = 'subplot',pause=False, click=True, save_plot=True)

def main():

    pdb.set_trace()

    return

if __name__ == "__main__":

    main()

    if len(self.local_elem) >= 2:

            # check if nearby peak is larger
        if len(self.local_elem) > 2:
            if peak_lat_sort[0] < peak_lat_sort[1]:
                stj_lat_result = self.STJ_lat_sort[1]
                'Second peak is greater - more than 2 peaks ID'
                pdb.set_trace()
            else:
                stj_lat_result = self.STJ_lat_sort[0]

        plot_method_compare = False
        if plot_method_compare:
            if self.STJ_lat_sort[0] != self.STJ_lat or len(self.local_elem) >= 2:
                if self.STJ_lat_sort[0] != self.STJ_lat:
                    print('Methods different')
                if len(self.local_elem) >= 2:
                    print('More than 2 jets')

    else:  # only 0 or 1 peaks
        if print_messages:
            print('single jet: ', self.STJ_lat_sort)

        if((hemi == 'SH' and self.STJ_lat_sort[0] < -40.0) or
           (hemi == 'NH' and self.STJ_lat_sort[0] > 40.0)):
            # when STJ is more likely the EDJ
            if hemi == 'SH':
                pdb.set_trace()

            save_file_testing = False
            if save_file_testing:

                # save data for Mike to test
                np.savez('{}/min_max_example.npz'.format(data_out_dir),
                         phi_2PV=self.phi_2PV, dtdphi_val=self.dtdphi_val)
                # test it opens
                npzfile = np.load('{}/min_max_example.npz')
                npzfile.files

def show_plot(pause, click):

    if pause:
        # pause sequence
        plt.draw()
        plt.pause(10)
        plt.close()
    else:
        if click:
            # button press sequence
            plt.draw()
            plt.waitforbuttonpress()
            plt.close()
        else:
             plt.show()
             plt.close()


def MakeOutputFile(filename, data, dim_name, var_name, var_type):

    f = io.netcdf.netcdf_file(filename, mode='w')
    for j in range(dim):
        f.createDimension(dim_name[j], len(data[dim_name[j]]))
    for i in range(len(var_name)):
        tmp = f.createVariable(var_name[i], var_type[i], var_dim_name[i])
        tmp[:] = data[var_name[i]]

    f.close()
    print('created file: ', filename)


def plot_u(plt,ax1,ax2,ax_cb,jet_NH,jet_SH,uwnd,var,t,t_elem,fname_out,save_plot):

        bounds = np.arange(-50, 51, 5)
        draw_map_model(plt, ax1, ax_cb, uwnd, var['lon'], var['lat'], '', '', 'BuRd',
                       bounds, None, True, domain=None, name_cbar=None,
                       coastline=True)

        # add jet positon to plots
        ax1.plot([0, 360], [jet_SH, jet_SH], c='#0033ff', linewidth=2)
        ax1.plot([0, 360], [jet_NH, jet_NH], c='#0033ff', linewidth=2)

        ax1.xaxis.tick_top()

        # plot u wind
        uzonal = MeanOverDim(data=uwnd, dim=1)
        # and the jet location
        ax2.plot([uzonal.min(), uzonal.max()], [jet_SH, jet_SH], c='#0033ff',
                 linewidth=2)
        ax2.plot([uzonal.min(), uzonal.max()], [jet_NH, jet_NH], c='#0033ff',
                     linewidth=2)

        ax2.set_ylim(-90, 90)
        ax2.plot(uzonal, var['lat'], c='k')
        fix_ax_label = gfdl_lat_change_map(ax=ax2)
        # ax2.set_ylim(-90,90)
        ax2.yaxis.tick_right()
        plt.setp(ax2.get_xticklabels(), visible=False)

        # set plot colour
        # ax2.patch.set_facecolor('#CCCCCC')

        plt.tight_layout()

        # reshape plots so on same horizontal
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        pos3 = ax_cb.get_position()

        new_height = pos1.height  # height = y1-y0
        y0 = pos1.y0 - 0.045

        # [Left,Bottom,Width,Hight]
        ax1.set_position([pos1.x0, y0, pos1.width, new_height])

        # [Left,Bottom,Width,Hight]
        ax2.set_position([pos2.x0 - 0.03, pos2.y0, pos2.width, pos2.height - 0.09])

        ax_cb.set_position([pos3.x0, pos3.y0 + 0.025, pos3.width -
                            0.182, pos3.height])  # [Left,Bottom,Width,Hight]
        ax_cb.set_xlabel(r'$\bar{u}$ $(ms^{-1})$')

        if save_plot :
          plt.savefig(fname_out)
          print 'Saved file: ', fname_out
          plt.show()
          plt.close()

        return

def make_u_plot(fname):

    #open jet data
    filename = '{}/STJ_data.nc'.format(data_out_dir)
    assert os.path.isfile(filename), 'File '+ filename +' does not exist. Need jet latitude for plotting.' 
    var_jet = openNetCDF4_get_data(filename)

    #open u wind data
    filename = '{}/u79_15.nc'.format(u_data_dir)
    assert os.path.isfile(filename), 'File '+ filename +' does not exist.' 
    var = openNetCDF4_get_data(filename)


    lev250 = FindClosestElem(25000, var['lev'])[0]

    #Which time elements are of interest?
    t_elem = [431,440]

    for t in range(len(t_elem)):


        uwnd = var['var131'][t_elem[t], lev250, :, :]
        # from running the code i know where the jet is. Plot it on a map as a
        # sanity check
        jet_NH = var_jet['STJ_lat'][t_elem[t],0]
        jet_SH = var_jet['STJ_lat'][t_elem[t],1]
        fname_out = '{}/uwind_{}_with_wind.eps'.format(plot_dir, t_elem[t])

        fig = plt.figure(figsize=(10, 5))

        # gs = mpl.gridspec.GridSpec(2,2, height_ratios=[0.4,0.1,0.2],
        #                            width_ratios=[4,1.25,0.1])    #wide[plot,plot,space]
        # gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93,x
        #           wspace=0.02, hspace=0.03)
        # ax1 = plt.subplot(gs[0])
        # ax2 = plt.subplot(gs[1])
        # ax_cb = plt.subplot(gs[2])

        # wind plot
        ax1 = plt.subplot2grid((10, 20), (0, 0), colspan=16, rowspan=9)
        # zonal mean
        ax2 = plt.subplot2grid((10, 20), (0, 16), colspan=5, rowspan=9)
        # colour bar
        ax_cb = plt.subplot2grid((10, 20), (9, 0), colspan=20)

        save_plot = True
        plot_u(plt,ax1,ax2,ax_cb,jet_NH,jet_SH,uwnd,var,t,t_elem,fname_out,save_plot) 
    pdb.set_trace()


def PlotCalendarTimeseries(STJ_cal_mean, STJ_cal_int_mean, STJ_cal_th_mean,
                           STJ_cal_x_mean, mean_val, PC):

    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    colour_mark = ['r', 'b']

    fig = plt.figure(figsize=(14, 8))
    # position
    ax1 = plt.subplot2grid((9, 6), (0, 0), colspan=6, rowspan=2)
    # intensity
    ax2 = plt.subplot2grid((9, 6), (2, 0), colspan=6, rowspan=2)
    # theta
    ax3 = plt.subplot2grid((9, 6), (4, 0), colspan=6, rowspan=2)
    # theta
    ax4 = plt.subplot2grid((9, 6), (6, 0), colspan=6, rowspan=2)

    # table
    ax5 = plt.subplot2grid((9, 6), (8, 0), colspan=6)

    ax1.set_xlim(0, 11)
    ax1.set_ylabel(r'$\phi_{STJ}$  ', rotation=0)
    ax2.set_xlim(0, 11)
    ax2.set_ylabel(r'$I (ms^{-1})$  ', rotation=0)
    ax3.set_xlim(0, 11)
    ax3.set_ylabel(r'$\theta (K)$  ', rotation=0)
    ax4.set_xlim(0, 11)
    ax4.set_ylabel(r'$\phi_x$  ', rotation=0)

    hemi = ['NH', 'SH']
    for hemi_count in range(2):
        # position
        ax1.plot(np.arange(0, 12, 1), np.abs(STJ_cal_mean[:, hemi_count]),
                 c=colour_mark[hemi_count], marker='x', markersize=8, linestyle='-')
        lat_mean = [mean_val['DJF', 'lat'][hemi_count],
                    mean_val['MAM', 'lat'][hemi_count],
                    mean_val['JJA', 'lat'][hemi_count],
                    mean_val['SON', 'lat'][hemi_count]]
        ax1.plot([0, 3, 6, 9], np.abs(lat_mean), c=colour_mark[hemi_count], marker='o',
                 markersize=8, linestyle=' ')

        # Intensity
        ax2.plot(np.arange(0, 12, 1), STJ_cal_int_mean[:, hemi_count],
                 c=colour_mark[hemi_count], marker='x', markersize=8, linestyle='-')
        I_mean = [mean_val['DJF', 'I'][hemi_count], mean_val['MAM', 'I'][hemi_count],
                  mean_val['JJA', 'I'][hemi_count], mean_val['SON', 'I'][hemi_count]]
        ax2.plot([0, 3, 6, 9], I_mean, c=colour_mark[hemi_count],
                 marker='o', markersize=8, linestyle=' ')

        ax3.plot(np.arange(0, 12, 1), STJ_cal_th_mean[:, hemi_count],
                 c=colour_mark[hemi_count], marker='x', markersize=8, linestyle='-')
        th_mean = [mean_val['DJF', 'th'][hemi_count], mean_val['MAM', 'th'][hemi_count],
                   mean_val['JJA', 'th'][hemi_count], mean_val['SON', 'th'][hemi_count]]
        ax3.plot([0, 3, 6, 9], th_mean, c=colour_mark[hemi_count],
                 marker='o', markersize=8, linestyle=' ')

        ax4.plot(np.arange(0, 12, 1), np.abs(STJ_cal_x_mean[:, hemi_count]),
                 c=colour_mark[hemi_count], marker='x', markersize=8, linestyle='-',
                 label=hemi[hemi_count])
        x_mean = [mean_val['DJF', 'x'][hemi_count], mean_val['MAM', 'x'][hemi_count],
                  mean_val['JJA', 'x'][hemi_count], mean_val['SON', 'x'][hemi_count]]
        ax4.plot([0, 3, 6, 9], np.abs(x_mean), c=colour_mark[hemi_count], marker='o',
                 markersize=8, linestyle=' ')

    # add season horizintal lines
    xx = np.arange(14)
    cut = (xx > 0) & (xx % 3 == 0)

    for x in xx[cut]:
        ax1.axvline(x=x - 1.5, ymin=-1.2, ymax=1, c="k", linestyle=':',
                    linewidth=1, zorder=0, clip_on=False)
        ax2.axvline(x=x - 1.5, ymin=-1.2, ymax=1.2, c="k",
                    linestyle=':', linewidth=1, zorder=0, clip_on=False)
        ax3.axvline(x=x - 1.5, ymin=-1.2, ymax=1.2, c="k",
                    linestyle=':', linewidth=1, zorder=0, clip_on=False)
        ax4.axvline(x=x - 1.5, ymin=0, ymax=1.2, c="k", linestyle=':',
                    linewidth=1, zorder=0, clip_on=False)

    # months as labels
    ax4.set_xticks(np.arange(0, 12, 1))
    ax4.set_xticklabels(months)

    plt.legend(loc=1)

    # turn off x axis labels on plots 1-2
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # var_name  = ['lat','int','lev','cross']

    # label1 = '         ' + r'$r_{\phi,I}=$'+ '{0:.2f}'.format(PC[0,1,1])
    # label2 = (r'$r_{\phi,\theta}=$'+ '{0:.2f}'.format(PC[0,2,0]) + '  ' +
    #           r'$r_{\phi,\theta}=$'+ '{0:.2f}'.format(PC[0,2,1]))
    # label3 = (r'$r_{\phi,x}=$'+ '{0:.2f}'.format(PC[0,3,0]) + '           ' +
    #           r'$r_{\phi,x}=$'+ '{0:.2f}'.format(PC[0,3,1]))

    # add the table to the plot
    celldata = [['{0:.2f}'.format(PC[0, 1, 0]), '{0:.2f}'.format(PC[0, 2, 0]),   # |
                 '{0:.2f}'.format(PC[0, 3, 0]), '{0:.2f}'.format(PC[1, 2, 0]),   # NH
                 '{0:.2f}'.format(PC[1, 3, 0]), '{0:.2f}'.format(PC[2, 3, 0])],  # |
                ['{0:.2f}'.format(PC[0, 1, 1]), '{0:.2f}'.format(PC[0, 2, 1]),   # -
                 '{0:.2f}'.format(PC[0, 3, 1]), '{0:.2f}'.format(PC[1, 2, 1]),   # SH
                 '{0:.2f}'.format(PC[1, 3, 1]), '{0:.2f}'.format(PC[2, 3, 1])]]  # -
    rowlabel = ['NH', 'SH']
    collabel = ['  ', r'$r_{\phi,I}$', r'$r_{\phi,\theta}$',
                r'$r_{\phi,x}$', r'$r_{I,\theta}$', r'$r_{I,x}$', r'$r_{\theta,x}$']
    table = latex_table(celldata, rowlabel, collabel)
    ax5.text(0.3, -.5, table, size=14)
    ax5.axis('off')

    plt.savefig('{}/calendar_mean.eps'.format(plot_dir))
    # plt.show()
    plt.close()
    hemi_count = hemi_count + 1

    # pdb.set_trace()

    fig = plt.figure(figsize=(15, 6))
    # position vs intensity
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    # intensity vs theta
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    # theta  va position
    ax3 = plt.subplot2grid((1, 3), (0, 2))

    ax1.set_xlabel(r'$\phi$  ')
    ax1.set_ylabel('I  ', rotation=0)
    ax2.set_xlabel('I  ')
    ax2.set_ylabel(r'$\theta$  ', rotation=0)
    ax3.set_xlabel(r'$\phi$  ')
    ax3.set_ylabel(r'$\theta$  ', rotation=0)

    for hemi_count in range(2):

        pos = (np.abs(STJ_cal_mean[:, hemi_count])).tolist()
        ints = STJ_cal_int_mean[:, hemi_count].tolist()
        th = STJ_cal_th_mean[:, hemi_count].tolist()

        pos.append(np.abs(STJ_cal_mean[0, hemi_count]))
        ints.append(STJ_cal_int_mean[0, hemi_count])
        th.append(STJ_cal_th_mean[0, hemi_count])

        ax1.plot(pos, ints, c=colour_mark[hemi_count],
                 marker='x', markersize=8, linestyle='-')
        ax2.plot(ints, th, c=colour_mark[hemi_count],
                 marker='x', markersize=8, linestyle='-')
        ax3.plot(pos, th, c=colour_mark[hemi_count], marker='x',
                 markersize=8, linestyle='-', label=hemi[hemi_count])

        # add months to each plot
        mm_count = 0

        for mm in months:
            ax1.annotate(mm, xy=(pos[mm_count], ints[mm_count] + 0.1),
                         ha='right', va='bottom', size=10, color=colour_mark[hemi_count])
            ax2.annotate(mm, xy=(ints[mm_count], th[mm_count] + 0.1),
                         ha='right', va='bottom', size=10, color=colour_mark[hemi_count])
            ax3.annotate(mm, xy=(pos[mm_count], th[mm_count] + 0.1), ha='right',
                         va='bottom', size=10, color=colour_mark[hemi_count])

            mm_count = mm_count + 1

    plt.legend(loc=2)
    plt.savefig('{}/calendar_lifecycle.eps'.format(plot_dir))
    # plt.show()
    plt.close()
    # pdb.set_trace()


def get_centred_bounds(min_bound, max_bound, increment, critical=None):

    if critical is None:

        bounds_low = np.arange(min_bound, 0, increment)
        bounds_high = np.arange(increment, max_bound + 0.1, increment)

    else:

        bounds_low = np.arange(min_bound, -math.floor(critical * 10) / 10 + .1, increment)
        bounds_high = np.arange(math.ceil(critical * 10) / 10 - .1, max_bound, increment)

    bounds = np.zeros(bounds_low.size + bounds_high.size)
    bounds[0:bounds_low.size] = bounds_low
    bounds[(bounds_low.size):] = bounds_high

    return bounds


def PlotPC_Matrix_subplot(ax1, ax1_position, ax_cb, matrix, var_name_latex):

    fontsize = 16
    bounds = get_centred_bounds(min_bound=-1.0, max_bound=1.0,
                                increment=0.1, critical=None)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'RdBu_cmap', ['Navy', 'white', 'Maroon'], N=len(bounds) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad(color='Gainsboro', alpha=1.)  # grey to Nan

    # can use either matshow or imshow
    cax = ax1.matshow(matrix, cmap=cmap, interpolation='nearest')
    ax1.set_position(ax1_position)

    # labels onto matrix
    ax1.set_xticks(range(len(var_name_latex)))
    ax1.set_xticklabels(var_name_latex, rotation=0, fontsize=fontsize)
    ax1.set_yticks(range(len(var_name_latex)))
    ax1.set_yticklabels(var_name_latex, fontsize=fontsize)

    # colour bar
    font = 18
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, ticks=bounds,
                                     spacing='uniform', orientation='horizontal',
                                     boundaries=bounds, format='%1.1g')


def PlotPC_matrix_single(matrix, var_name_latex, filename):

    fig = plt.figure(1, figsize=(8, 9))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1_position = [0.1, 0.1, 0.88, 0.95]
    ax_cb = fig.add_axes([0.12, 0.1, 0.8, 0.05])

    PlotPC_Matrix_subplot(ax1, ax1_position, ax_cb, matrix, var_name_latex)

    plt.savefig(filename)
    # plt.show()

    plt.close()


def PlotPC(Annual, Seasonal, Monthly, var_name, diri):

    var_name_latex = [r'$\phi$', r'I', r'$\theta$', r'x']

    # prep storage matrix
    matrix = np.zeros([4, 4])
    tmp = np.tri(matrix.shape[0], k=-1)
    tmp[np.where(tmp == 1)] = np.nan  # lower triangular as nan
    matrix = tmp + matrix

    matrix_NH = copy.deepcopy(matrix)
    matrix_SH = copy.deepcopy(matrix)

    matrix_NH = Annual[:, :, 0] + matrix
    matrix_SH = Annual[:, :, 1] + matrix

    path = diri.plot_loc

    # Annual plots
    PlotPC_matrix_single(matrix_NH, var_name_latex, filename=path + 'NH_PC_matrix.eps')
    PlotPC_matrix_single(matrix_SH, var_name_latex, filename=path + 'SH_PC_matrix.eps')

    # SeasonalPlots
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        hemi_count = 0
        for hemi in ['NH', 'SH']:
            print(hemi, '', season)
            matrix_season = copy.deepcopy(matrix)
            plot_matrix = Seasonal[season][:, :, hemi_count] + matrix_season
            PlotPC_matrix_single(plot_matrix, var_name_latex,
                                 filename=path + hemi + '_PC_matrix_' + season + '.eps')
            hemi_count = hemi_count + 1

    #pdb.set_trace()
