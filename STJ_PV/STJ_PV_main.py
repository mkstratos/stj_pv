import pdb
import os
import collections
import platform

# Dependent code
from general_functions import openNetCDF4_get_data
from MakeIPV_data import Generate_IPV_Data
import STJ_IPV_metric
from IPV_plots import plot_u


# File purpose:  Calculate the subtropical jet position using the 2PV contour.
__author__ = "Penelope Maher"

# Created two named tuples to manage storage of index
data_name = collections.namedtuple('data_name', 'letter label')
metric = collections.namedtuple('metric', 'name hemisphere intensity position')


class Directory:
    'Working directory information and set precision'

    def __init__(self):
        """Code assumes that environment variables are assigned in each run location.
         work_loc:  Which server or computer you are currently working from
         data_loc:  Where is the data stored
        """
        # To set up an env var in the .bashrc add the following
        # export BASE=path_to_dir/

        self.base = os.environ['BASE']

        if self.base == '/home/maher/Documents/Penny':
            self.work_loc = 'PeronalLaptop'
            self.data_loc = '/media/Seagate Expansion Drive/Data/'

        if self.base == '/home/pm366/Documents':
            self.work_loc = 'ExeterLaptop'
            self.data_loc = '/media/pm366/Seagate Expansion Drive/Data/'

        if self.base == '/home/links/pm366/Documents/':
            self.work_loc = 'gv'
            self.data_loc = '/scratch/pm366/'

        self.plot_loc = self.base + '/Plots/'

        # Test if the string self.work_loc has been assigned
        assert isinstance(self.work_loc, str), ('Unknown base environment variable.'
                                                ' SetDefault.py')


class Experiment(object):
    'Assign run options and set paths to data.'

    def __init__(self, diri):

        self.diri = diri

        # Save: Run and save data. Open: Open data from previous run
        # RunNotSave: run code but do not save output
        RunFlag = ['Open', 'Save', 'RunNotSave']
        self.RunOpt = RunFlag[0]

        # Using daily or monthly data? Code designed around monthly data
        time_unit_opt = ['dd', 'mm']
        self.time_units = time_unit_opt[1]

        # Flag options for different data use. In this case model output or ERA data
        data_options = ['GFDL', 'Era']
        self.data_type = data_options[1]

        # Set true to check if data exists in location specified in file_loc
        skip_server_warning = True
        file_loc = 'emps-gv1.ex.ac.uk'
        if not skip_server_warning:
            assert platform.node() == file_loc, 'Data is not on ' + platform.node()

    def PathFilenameERA(self):

        if self.time_units == 'mm':
          time_syn = '_monthly.nc'
        else:
          time_syn = '_daily.nc'
 
        path = "{}{}".format(self.diri.data_loc, 'Data/ERA_INT/1979_2015/')
        self.u_fname = "{}{}".format(path, 'u79_15.nc')
        self.v_fname = "{}{}".format(path, 'v79_15.nc')
        self.t_fname = "{}{}".format(path, 't79_15.nc')


        # used a named tuple to manage the file variable label
        # (what the .nc file calls it) and
        # working letter (what this code will call the variable)
        self.var_names = {'t': data_name(letter='t', label='t'),
                          'u': data_name(letter='u', label='var131'),
                          'v': data_name(letter='v', label='var132'),
                          'p': data_name(letter='p', label='lev')}

        # Data window to use
        self.start_time = 0  # first month of data to use from file
        self.end_time = 360  # last month of data to use from file +1


def main():

    # Set data paths and working location
    diri = Directory()
    # Set filenames and run specifics like data type
    Exp = Experiment(diri)

    if Exp.data_type == 'Era':
        Exp.PathFilenameERA()
    else:
        print('Currently code only works for Era-Int data format')
        pdb.set_trace()

    plot_u_wind = False
    if plot_u_wind:
        plot_u(Exp.u_fname)

    file_type_opt = ['.nc', '.p']          # nc file or pickle
    file_type = file_type_opt[0]
    fileIPV_1 = Exp.path + 'IPV_data_79_16'      # IPV every 5K between 300-500
    fileIPV_2 = Exp.path + 'IPV_data_u_H_79_16'  # If using pickle this file is not needed

    if (Exp.RunOpt == 'Save') or (Exp.RunOpt == 'RunNotSave'):

        # init the object
        STJ_PV = Generate_IPV_Data(Exp)
        # Open the data
        STJ_PV.OpenFile()
        # Calculate the thermal definition of tropopause height
        STJ_PV.GetThermalTropopause()
        STJ_PV.GetIPV()

        if Exp.RunOpt == 'Save':
            # output data for faster read in
            STJ_PV.SaveIPV(fileIPV_1, fileIPV_2, file_type)
    else:  # if open
        STJ_PV = Generate_IPV_Data(Exp)
        STJ_PV.open_ipv_data(fileIPV_1, fileIPV_2, file_type)

        # Now that IPV has been calculated - calculate the STJ metric
        STJ_IPV_metric.calc_metric(STJ_PV.IPV_data)

        pdb.set_trace()
        # STJ_NH, STJ_SH = STJ_PV.Get_uwind_strength()

        # Save results
        # STJ_Post_Proc = STJ_Post_Processing()
        # STJ_Post_Proc.SaveSTJMetric(STJ_NH, STJ_SH)
        # STJ_Post_Proc.PlotIPV(output_plotting)

        filename = Exp.path + 'STJ_metric.nc'
        var = openNetCDF4_get_data(filename)
        pdb.set_trace()



    return ()


if __name__ == "__main__":

    main()
