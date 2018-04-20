# -*- coding: utf-8 -*-
"""
Run STJ: Main module "glue" that connects Subtropical Jet Metric calc, plot and diags.

To run, set stj configuration file, start and end dates in `main()` and run with
`$ python run_stj.py`

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


import pdb

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
            for var in ['uwnd', 'vwnd', 'tair', 'omega']:   #TODO: Need to change list
                if var not in self.data_cfg['file_paths']:
                    # This replicates the path in 'all' so each variable points to it
                    # this allows for the same loop no matter if data is in multiple files
                    self.data_cfg['file_paths'][var] = self.data_cfg['file_paths']['all']

        if 'wpath' not in self.data_cfg:
            # Sometimes, can't write to original data's path, so wpath is set
            # if it isn't, then wpath == path is fine, set that here
            self.data_cfg['wpath'] = self.data_cfg['path']

        self._set_metric()
        self.log_setup()

    def _set_metric(self):
        """Set metric and associated levels."""
        if self.config['method'] == 'STJPV':
            self.th_levels = np.array([265.0, 275.0, 285.0, 300.0, 315.0, 320.0, 330.0,
                                       350.0, 370.0, 395.0, 430.0])
            self.metric = stj_metric.STJPV
        elif self.config['method'] == 'STJUMax':
            self.p_levels = np.array([1000., 925., 850., 700., 600., 500., 400., 300.,
                                      250., 200., 150., 100., 70., 50., 30., 20., 10.])
            self.metric = stj_metric.STJMaxWind
        elif self.config['method'] == 'KangPolvani':
            self.metric = stj_metric.STJKangPolvani
        else:
            self.metric = None

    def _set_output(self, date_s=None, date_e=None):

        if self.config['method'] == 'STJPV':
            self.config['output_file'] = ('{short_name}_{method}_pv{pv_value}_'
                                          'fit{fit_deg}_y0{min_lat}'
                                          .format(**dict(self.data_cfg, **self.config)))

            self.th_levels = np.array([265.0, 275.0, 285.0, 300.0, 315.0, 320.0, 330.0,
                                       350.0, 370.0, 395.0, 430.0])
            self.metric = stj_metric.STJPV

        elif self.config['method'] == 'STJUMax':
            self.config['output_file'] = ('{short_name}_{method}_pres{pres_level}'
                                          '_y0{min_lat}'
                                          .format(**dict(self.data_cfg, **self.config)))

            self.p_levels = np.array([1000., 925., 850., 700., 600., 500., 400., 300.,
                                      250., 200., 150., 100., 70., 50., 30., 20., 10.])
            self.metric = stj_metric.STJMaxWind

        elif self.config['method'] == 'KangPolvani':

            self.config['output_file'] = ('{short_name}_{method}'
                                          .format(**dict(self.data_cfg, **self.config)))
            self.metric = stj_metric.STJKangPolvani

        else:
            self.config['output_file'] = ('{short_name}_{method}'
                                          .format(**dict(self.data_cfg, **self.config)))
            self.metric = None

        if date_s is not None and isinstance(date_s, dt.datetime):
            self.config['output_file'] += '_{}'.format(date_s.strftime('%Y-%m-%d'))

        if date_e is not None and isinstance(date_e, dt.datetime):
            self.config['output_file'] += '_{}'.format(date_e.strftime('%Y-%m-%d'))

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

    def _get_data(self, date_s=None, date_e=None):
        """Retrieve data stored according to `self.data_cfg`."""
        if self.config['method'] == 'STJPV':
            data = inp.InputData(self, date_s, date_e)
        elif self.config['method'] == 'STJUMax':
            data = inp.InputDataWind(self, ['uwnd'], date_s, date_e)
        else:
            data = inp.InputDataWind(self, ['uwnd', 'vwnd'], date_s, date_e)

        data.get_data_input()
        return data

    def run(self, date_s=None, date_e=None):
        """
        Find the jet, save location to a file.

        Parameters
        ----------
        date_s, date_e : :class:`datetime.datetime`
            Beginning and end dates, optional. If not included,
            use (Jan 1, self.year_s) and/or (Dec 31, self.year_e)

        """
        if date_s is None:
            date_s = dt.datetime(self.config['year_s'], 1, 1)
        if date_e is None:
            date_e = dt.datetime(self.config['year_e'], 12, 31)

        self._set_output(date_s, date_e)

        if self.data_cfg['single_year_file']:
            for year in range(date_s.year, date_e.year + 1):
                _date_s = dt.datetime(year, 1, 1)
                _date_e = dt.datetime(year, 12, 31)
                self.log.info('FIND JET FOR %s - %s', _date_s.strftime('%Y-%m-%d'),
                              _date_e.strftime('%Y-%m-%d'))
                data = self._get_data(_date_s, _date_e)
                jet = self.metric(self, data)

                for shemis in [True, False]:
                    jet.find_jet(shemis)

                if year == date_s.year:
                    jet_all = jet
                else:
                    jet_all.append(jet)
        else:
            data = self._get_data(date_s, date_e)
            jet_all = self.metric(self, data)
            for shemis in [True, False]:
                jet_all.find_jet(shemis)

        jet_all.save_jet()

    def run_sensitivity(self, sens_param, sens_range, date_s=None, date_e=None):
        """
        Perform a parameter sweep on a particular parameter of the JetFindRun.

        Parameters
        ----------
        sens_param : string
            Configuration parameter of :py:meth:`~STJ_PV.run_stj.JetFindRun`
        sens_range : iterable
            Range of values of `sens_param` over which to iterate
        date_s, date_e : :class:`datetime.datetime`, optional
            Start and end dates, respectively. Optional, defualts to config file defaults

        """
        params_avail = ['fit_deg', 'pv_value', 'min_lat']
        if sens_param not in params_avail:
            print('SENSITIVITY FOR {} NOT AVAILABLE'.format(sens_param))
            print('POSSIBLE PARAMS:')
            for param in params_avail:
                print(param)
            sys.exit(1)

        for param_val in sens_range:
            self.log.info('----- RUNNING WITH %s = %f -----', sens_param, param_val)
            self.config[sens_param] = param_val
            self._set_output(date_s, date_e)
            self.log.info('OUTPUT TO: %s', self.config['output_file'])
            self.run(date_s, date_e)


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
        config = yaml.safe_load(cfg.read())

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

    # When either `missing` or `wrong_type` have values, this will evaluate `True`
    if missing or wrong_type:
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
        if config['method'] not in ['STJPV', 'STJUMax', 'KangPolvani']:
            # config must have pfac if it's pressure level data
            missing_optionals.append(False)
            print('NO METHOD FOR HANDLING: {}'.format(config['method']))

        elif config['method'] == 'STJPV':
            opt_keys = {'poly': str, 'fit_deg': int, 'pv_value': float}
            _, missing_opt = check_config_req(cfg_file, opt_keys, id_file=False)
            missing_optionals.append(missing_opt)

        elif config['method'] == 'STJUMax':
            opt_keys = {'pres_level': float, 'min_lat': float}
            _, missing_opt = check_config_req(cfg_file, opt_keys, id_file=False)
            missing_optionals.append(missing_opt)
        elif config['method'] == 'KangPolvani':
            opt_keys = {'pres_level': float}
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

    #jf_run = JetFindRun('./conf/stj_kp_erai_daily_gv.yml')
    #jf_run = JetFindRun('./conf/stj_config_merra_daily.yml')
    #jf_run = JetFindRun('./conf/stj_config_ncep_monthly.yml')
    jf_run = JetFindRun('./conf/stj_config_erai_theta_daily.yml')
    date_s = dt.datetime(1979, 1, 1)
    date_e = dt.datetime(2016, 12, 31)

    jf_run.run(date_s, date_e)
    # jf_run.run_sensitivity(sens_param='pv_value', sens_range=np.arange(1.0, 4.5, 0.5),
    #                        year_s=1979, year_e=2016)
    jf_run.log.info('JET FINDING COMPLETE')


if __name__ == "__main__":
    main()
