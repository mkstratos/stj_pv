"""
Run STJ: Main module "glue" that connects Subtropical Jet Metric calc, plot and diags.

Authors: Penelope Maher, Michael Kelleher
"""
import os
import logging
import datetime as dt
import collections
import numpy as np
import yaml
import stj_metric
import input_data as inp
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
                             'pv_value': 2.0, 'start_time': 0, 'end_time': None,
                             'dlat': 0.2, 'dtheta': 1.0, 'nslice': 8, 'freq': 'monthly',
                             'log_file': './{}_stj_find.log'.format(dt.datetime.now())}

            self.plot_opts = {'debug': False, 'metric': True, 'extn': 'png'}
        self.log = self.log_setup

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

    def log_setup(self):
        """Create a logger object with file location from `self.run_opts`."""

        logger = logging.getLogger('stjfind')
        logger.setLevel(logging.DEBUG)

        log_file_handle = logging.FileHandler(self.run_opts['log_file'])
        log_file_handle.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file_handle.setFormatter(formatter)

        logger.addHandler(log_file_handle)
        return logger


class JetFindRun(object):
    """
    Class containing properties about an individual attempt to find the subtropical jet.

    Attributes
    ----------
    data_source : string
        Path to input data configuration file
    config : dict
        Dictionary of properties of the run
    freq : Tuple
        Output data frequency (time, spatial)
    method : STJMetric
        Jet finder type
    log : logger
        Debug log

    Methods
    -------
    setup_logger
    find_jet
    write_data
    """

    def __init__(self, config_file=None):
        """
        Initialise jet finding attempt.

        Parameters
        ----------
        config : string, optional
            Location of YAML-formatted configuration file, default None
        """
        now = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if config_file is None:
            # Use default parameters if none are specified
            self.config = {'data_cfg': './data_config_default.yml', 'freq': 'mon',
                           'method': 'STJPV', 'log_file': "stj_find_{}.log".format(now),
                           'zonal_opt': 'mean', 'poly': 'cheby',
                           'pv_value': 2.0, 'fit_deg': 12, 'min_lat': 10.0,
                           'update_pv': False, 'year_s': 1979, 'year_e': 2015}
        else:
            # Open the configuration file, put its contents into a variable to be read by
            # YAML reader
            with open(config_file) as cfg:
                self.config = yaml.load(cfg.read())

            if '{}' in self.config['log_file']:
                # Log file name contains a format placeholder, use current time
                self.config['log_file'] = self.config['log_file'].format(now)

        with open(self.config['data_cfg']) as data_cfg:
            self.data_cfg = yaml.load(data_cfg.read())

        if self.data_cfg['single_var_file']:
            for var in ['uwnd', 'vwnd', 'tair']:
                if var not in self.data_cfg['file_paths']:
                    # This replicates the path in 'all' so each variable points to it
                    # this allows for the same loop no matter if data is in multiple files
                    self.data_cfg['file_paths'][var] = self.data_cfg['file_paths']['all']


        if self.config['method'] == 'STJPV':
            self.config['output_file'] = ('{short_name}_{method}_pv{pv_value}_'
                                          'fit{fit_deg}_y0{min_lat}'
                                          .format(**self.data_cfg, **self.config))

            self.th_levels = np.array([265.0, 275.0, 285.0, 300.0, 315.0, 320.0, 330.0,
                                       350.0, 370.0, 395.0, 430.0])
            self.metric = stj_metric.STJPV

        else:
            self.config['output_file'] = ('{short_name}_{method}'
                                          .format(**self.data_cfg, **self.config))
            self.metric = None

        self.log_setup()

    def log_setup(self):
        """Create a logger object with file location from `self.config`."""

        logger = logging.getLogger(self.config['method'])
        logger.setLevel(logging.DEBUG)

        log_file_handle = logging.FileHandler(self.config['log_file'])
        log_file_handle.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file_handle.setFormatter(formatter)

        logger.addHandler(log_file_handle)
        self.log = logger

    def _get_data(self, curr_year=None):
        """Retrieve data stored according to `self.data_cfg`."""
        data = inp.InputData(self, curr_year)
        data.get_data_input()
        return data

    def run(self, year_s, year_e):

        for year in range(year_s, year_e + 1):
            data = self._get_data(year)
            jet = self.metric(self, data)

            for shemis in [True, False]:
                jet.find_jet(shemis)

            if year == year_s:
                jet_all = jet
            else:
                jet_all.append(jet)

        jet_all.save_jet()

def main():
    """Main method, run STJ Metric."""

    # Generate an STJProperties, allows easy access to these properties across methods.
    stj_props = STJProperties()
    print(stj_props)
    stj_metric = metric.STJIPVMetric(stj_props)


if __name__ == "__main__":
    main()
