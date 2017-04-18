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
