# -*- coding: utf-8 -*-
"""
Run STJ: Main module "glue" that connects Subtropical Jet Metric calc, plot and diags.

Authors: Penelope Maher, Michael Kelleher
"""
import sys
import logging
import datetime as dt
import warnings
import numpy as np
import yaml
import stj_metric
import input_data as inp


np.seterr(all='ignore')
warnings.simplefilter('ignore', np.polynomial.polyutils.RankWarning)


class JetFindRun(object):
    """
    Class containing properties about an individual attempt to find the subtropical jet.

    Parameters
    ----------
    config : string, optional
        Location of YAML-formatted configuration file, default None

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
    :py:meth:`~log_setup`
    :py:meth:`~run`

    """

    def __init__(self, config_file=None):
        """Initialise jet finding attempt."""
        now = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if config_file is None:
            # Use default parameters if none are specified
            self.config = {'data_cfg': './conf/data_config_default.yml', 'freq': 'mon',
                           'method': 'STJPV', 'log_file': "stj_find_{}.log".format(now),
                           'zonal_opt': 'mean', 'poly': 'cheby',
                           'pv_value': 2.0, 'fit_deg': 12, 'min_lat': 10.0,
                           'update_pv': False, 'year_s': 1979, 'year_e': 2015}
        else:
            # Open the configuration file, put its contents into a variable to be read by
            # YAML reader
            self.config, cfg_failed = check_run_config(config_file)
            if cfg_failed:
                print('CONFIG CHECKS FAILED...EXITING')
                sys.exit(1)

            if '{}' in self.config['log_file']:
                # Log file name contains a format placeholder, use current time
                self.config['log_file'] = self.config['log_file'].format(now)

        self.data_cfg, data_cfg_failed = check_data_config(self.config['data_cfg'])
        if data_cfg_failed:
            print('DATA CONFIG CHECKS FAILED...EXITING')
            sys.exit(1)

        if self.data_cfg['single_var_file']:
            for var in ['uwnd', 'vwnd', 'tair', 'omega']:
                if var not in self.data_cfg['file_paths']:
                    # This replicates the path in 'all' so each variable points to it
                    # this allows for the same loop no matter if data is in multiple files
                    self.data_cfg['file_paths'][var] = self.data_cfg['file_paths']['all']

        if 'wpath' not in self.data_cfg:
            # Sometimes, can't write to original data's path, so wpath is set
            # if it isn't, then wpath == path is fine, set that here
            self.data_cfg['wpath'] = self.data_cfg['path']

        if self.config['method'] == 'STJPV':
            self.config['output_file'] = ('{short_name}_{method}_pv{pv_value}_'
                                          'fit{fit_deg}_y0{min_lat}'
                                          .format(**dict(self.data_cfg, **self.config)))

            self.th_levels = np.array([265.0, 275.0, 285.0, 300.0, 315.0, 320.0, 330.0,
                                       350.0, 370.0, 395.0, 430.0])
            self.metric = stj_metric.STJPV

        else:
            self.config['output_file'] = ('{short_name}_{method}'
                                          .format(**dict(self.data_cfg, **self.config)))
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

    def run(self, year_s=None, year_e=None):
        """
        Find the jet, save location to a file.

        Parameters
        ----------
        year_s, year_e : int
            Beginning and end years, optional. If not included,
            use self.year_s and/or self.year_e

        """
        if year_s is None:
            year_s = self.config['year_s']
        if year_e is None:
            year_e = self.config['year_e']

        if self.data_cfg['single_year_file']:
            for year in range(year_s, year_e + 1):
                self.log.info('FIND JET FOR %d', year)
                data = self._get_data(year)
                jet = self.metric(self, data)

                for shemis in [True, False]:
                    jet.find_jet(shemis)

                if year == year_s:
                    jet_all = jet
                else:
                    jet_all.append(jet)
        else:
            data = self._get_data(year_s)
            jet_all = self.metric(self, data)
            for shemis in [True, False]:
                jet_all.find_jet(shemis)

        jet_all.save_jet()


def check_config_req(cfg_file, required_keys_all, id_file=True):
    """
    Check that required keys exist within a configuration file.

    Parameters
    ----------
    cfg_file : string
        Path to configuration file
    required_keys_all : list
        Required keys that must exist in configuration file

    Returns
    -------
    config : dict
        Dictionary of loaded configuration file
    mkeys : bool
        True if required keys are missing

    """
    with open(cfg_file) as cfg:
        config = yaml.load(cfg.read())

    if id_file:
        print('{0} {1:^40s} {0}'.format(7 * '#', cfg_file))
    keys_in = config.keys()
    missing = []
    wrong_type = []
    for key in required_keys_all:
        if key not in keys_in:
            missing.append(key)
            check_str = u'[\U0001F630  MISSING]'
        elif not isinstance(config[key], required_keys_all[key]):
            wrong_type.append(key)
            check_str = u'[\U0001F621  WRONG TYPE]'
        else:
            check_str = u'[\U0001F60E  OKAY]'

        print(u'{:30s} {:30s}'.format(key, check_str))

    if len(missing) > 0 or len(wrong_type) > 0:
        print(u'{} {:2d} {:^27s} {}'.format(12 * '>', len(missing) + len(wrong_type),
                                            'KEYS MISSING OR WRONG TYPE', 12 * '<'))

        for key in missing:
            print(u'    MISSING: {} TYPE: {}'.format(key, required_keys_all[key]))
        for key in wrong_type:
            print(u'    {} ({}) IS WRONG TYPE SHOULD BE {}'.format(key, type(config[key]),
                                                                   required_keys_all[key]))
        mkeys = True
    else:
        mkeys = False

    return config, mkeys


def check_run_config(cfg_file):
    """
    Check the settings in a run configuration file.

    Parameters
    ----------
    cfg_file : string
        Path to configuration file

    """
    required_keys_all = {'data_cfg': str, 'freq': str, 'zonal_opt': str, 'method': str,
                         'log_file': str, 'year_s': int, 'year_e': int}

    config, missing_req = check_config_req(cfg_file, required_keys_all)

    # Optional checks
    missing_optionals = []
    if not missing_req:
        if config['method'] not in ['STJPV']:
            # config must have pfac if it's pressure level data
            missing_optionals.append(False)
            print('NO METHOD FOR HANDLING: {}'.format(config['method']))

        elif config['method'] == 'STJPV':
            opt_keys = {'poly': str, 'fit_deg': int, 'pv_value': float}
            _, missing_opt = check_config_req(cfg_file, opt_keys, id_file=False)
            missing_optionals.append(missing_opt)

    return config, any([missing_req, all(missing_optionals)])


def check_data_config(cfg_file):
    """
    Check the settings in a data configuration file.

    Parameters
    ----------
    cfg_file : string
        Path to configuration file

    """
    required_keys_all = {'path': str, 'short_name': str, 'single_var_file': bool,
                         'single_year_file': bool, 'file_paths': dict, 'pfac': float,
                         'lon': str, 'lat': str, 'lev': str, 'time': str, 'ztype': str}

    config, missing_req = check_config_req(cfg_file, required_keys_all)
    # Optional checks
    missing_optionals = []
    if not missing_req:
        if config['ztype'] == 'pres':
            # config must have pfac if it's pressure level data
            opt_reqs = {'pfac': float}
            _, miss_opts = check_config_req(cfg_file, opt_reqs)
            missing_optionals.append(miss_opts)

        elif config['ztype'] not in ['pres', 'theta']:
            print('NO METHOD TO HANDLE {} level data'.format(config['ztype']))
            missing_optionals.append(True)
        else:
            missing_optionals.append(False)
    return config, any([missing_req, all(missing_optionals)])


def main():
    """Run the STJ Metric given a configuration file."""
    # Generate an STJProperties, allows easy access to these properties across methods.
    jf_run = JetFindRun('./conf/stj_config_erai_monthly_gv.yml')
    #jf_run = JetFindRun('./conf/stj_config_erai_theta.yml')
    jf_run.run(1979, 2016)
    jf_run.log.info('JET FINDING COMPLETE')

if __name__ == "__main__":
    main()
