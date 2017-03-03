"""
Run STJ: Main module "glue" that connects Subtropical Jet Metric calc, plot and diags.

Authors: Penelope Maher, Michael Kelleher
"""
import os
import collections
import numpy as np
import stj_metric as metric
#import input_data as makeipv

# Metric named tuple, use later?
# metric = collections.namedtuple('metric', 'name hemisphere intensity position')


class STJProperties(object):
    """
    Object containing properties about Subtropical Jet Metric calculation.

    Attributes
    ----------
    base : Directory containing output data and plotting directories are
    data_source : What has generated data to be used (e.g. ERA-Interim, NCEP/NCAR
        reanalysis, GFDL idealised model...
    directories : Dictionary of locations, keys are 'work_loc' and 'data_loc'
        work_loc is what machine you're running from, data_loc contains input data
    in_files : Dictionary of sub-directory containing specific files,
        and file names for isobaric u-wind, v-wind, and air temperature
    var_names : Names of netCDF variables within each file (u, v, t)
    run_opts : Dictionary of run-time options (See below)
    plot_opts : Dictionary of plotting options

    Methods
    -------
    find_location
    set_file_and_var_names
    create_output_dirs
    run

    Notes
    -----
    **Run options** `run_opts`

    - `run_flag`: What mode should the metric be run on? Either:
        - 'save' (Calculate IPV, run metric, then save)
        - 'open' (Load stored IPV, calculate metric),
        - 'runnotsave' (Calculate IPV, run metric, don't save IPV output)

    - `slicing`: 'zonal_mean' or 'indv' for zonal mean position, or position at
        each longitude

    - `debug`: Print debug statements, include set_trace() for True

    - `pv_value`: Get metric using this value of isentropic pv (Default = 2.0 PVU, float)

    - `start_time`: Time index of first metric calculation (integer)

    - `end_time`: Time index of last metric calculation (integer)

    - `dlat`: Spacing between latitude points for interpolation

    **Plotting options** `plot_opts`
        - `debug`: Boolean if True, make debugging plots
        - `metric`: Make plots locating the jet, and identifying its intensity
        - `extn`: File extension (e.g.: eps, pdf, png)
    """
    def __init__(self, conf_file=None):
        """
        Initialise default properties for subtropical jet metric.
        """

        try:
            self.base = os.environ['BASE']
        except KeyError:
            print('Env var BASE not set, using defaults')
            self.base = './'

        self.data_source = 'ERA-INT'
        self.directories = None
        self.in_files = None
        self.var_names = None
        self.coord_names = None

        self.find_location()
        self.set_file_and_var_names()
        if conf_file is None:
            self.th_levels = np.arange(300, 501, 5)
            self.run_opts = {'run_flag': 'save', 'slicing': 'zonal_mean', 'debug': True,
                             'pv_value': 2.0, 'start_time': 0, 'end_time': 360,
                             'dlat': 0.2, 'dtheta': 1.0, 'nslice': 8}
            self.plot_opts = {'debug': False, 'metric': True, 'extn': 'png'}

    def __str__(self):
        """
        Method to pretty-print information about a particular STJ Metric calculation.
        Called on `print(STJProperties)`
        """
        out_str = "{0} STJ Metric Properties {0}\n".format('-' * 5)
        n_header = len(out_str) - 1
        out_str += 'Data Source: {}\n'.format(self.data_source)
        for opt in self.run_opts:
            out_str += '   {:12s} : {}\n'.format(opt, self.run_opts[opt])
        for opt in self.plot_opts:
            out_str += '   {:12s} : {}\n'.format(opt, self.plot_opts[opt])
        out_str += '-' * n_header
        return out_str

    def find_location(self):
        """
        Find and assign locations for input/output data and plots based on `self.base`.

        Uses dictionary that defines locations.
        All that needs to be added is a new key for each new base, with a dictionary
        for it containing the 'work_loc' and 'data_loc'

        Use current directory (./) and (./Data) as the defaults (from __init__ if env
        variable $BASE isn't set.
        """

        dir_locs = {'/home/maher/Documents/Penny':
                    {'work_loc': 'PeronalLaptop',
                     'data_loc': '/media/Seagate Expansion Drive/Data/'},

                    '/home/pm366/Documents':
                    {'work_loc': 'ExeterLaptop',
                     'data_loc': '/media/pm366/Seagate Expansion Drive/Data/'},

                    '/home/links/pm366/Documents/':
                    {'work_loc': 'gv',
                     'data_loc': '/scratch/pm366/'},

                    '/home/links/mk450/stj_pv':
                    {'work_loc': 'gv',
                     'data_loc': '/scratch/mk450/'},

                    '/Users/mk450/stj_pv':
                    {'work_loc': 'MKiMac',
                     'data_loc': '/Volumes/FN_2187/erai/'},

                    './': {'work_loc': 'LocalHost', 'data_loc': './Data'}}

        self.directories = dir_locs[self.base]
        self.directories['plot_loc'] = '{}/Plots'.format(self.base)

    def set_file_and_var_names(self):
        """
        Set filenames and variables within those files depending on `self.data_source`.
        """

        # Initalize a Named Tuple to store how variables are called within the input files
        data_name = collections.namedtuple('data_name', 'letter label')

        # Currently, only ERA-INT is setup to work, but this file_dict can be expanded
        # To include a data type, it's input directory, and locations of u/v/t files
        # Also, include the input/output file for IPV
        file_dict = {'ERA-INT':
                     {'in_dir': '{}/{}'.format(self.directories['data_loc'],
                                               'Data/ERA_INT/1979_2015/'),
                      'u': 'u79_15.nc', 'v': 'v79_15.nc', 't': 't79_15.nc',
                      'ipv1': 'IPV_data_79_15.nc', 'ipv2': 'IPV_data_u_H_79_15.nc'}}

        if self.data_source == 'ERA-INT':
            self.var_names = {'t': data_name(letter='t', label='t'),
                              'u': data_name(letter='u', label='var131'),
                              'v': data_name(letter='v', label='var132'),
                              'p': data_name(letter='p', label='lev')}
            self.coord_names = {'time': 'time', 'lev': 'lev', 'lat': 'lat', 'lon': 'lon'}
        self.in_files = file_dict[self.data_source]

    def create_output_dirs(self):
        """
        Create output directories.
        """
        for dir_type in self.directories:
            if not os.path.exists(self.directories[dir_type]):
                print('CREATING {} DIRECTORY: {}'.format(dir_type,
                                                         self.directories[dir_type]))
                os.makedirs(self.directories[dir_type])

    def run(self):
        """
        Run STJ Metric with properties defined within `self`
        """


def main():
    """
    Main method, run STJ Metric
    """

    # Generate an STJProperties, allows easy access to these properties across methods.
    stj_props = STJProperties()
    print(stj_props)
    stj_metric = metric.STJIPVMetric(stj_props)


if __name__ == "__main__":
    main()
