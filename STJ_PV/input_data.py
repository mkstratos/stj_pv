# -*- coding: utf-8 -*-
"""Generate or load input data for STJ Metric."""
import os
import numpy as np
import netCDF4 as nc
# Dependent code
import utils
import data_out as dout
import psutil
# if stream function remains in technique then copy over. For now just read it in
# /home/pm366/Documents/ShareCode/Atmosphere/StreamFunction.py
from StreamFunction import cal_stream_fn

import matplotlib.pyplot as plt
from CommonFunctions import FindClosestElem

__author__ = "Penelope Maher, Michael Kelleher"


class InputData(object):
    """
    Contains the relevant input data and routines for an JetFindRun.

    Parameters
    ----------
    jet_find : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Object containing properties about the metric calculation to be done. Used to
        locate correct files, and variables within those files.
    year : int, optional
        Year of data to load, not used when all years are in a single file

    """

    def __init__(self, props, year=None):
        """Initialize InputData object, using JetFindRun class."""
        self.props = props
        self.config = props.config
        self.data_cfg = props.data_cfg
        self.year = year

        # Initialize attributes defined in open_files or open_ipv_data
        self.time = None
        self.time_units = None
        self.calendar = None

        self.lon = None
        self.lat = None
        self.lev = None
        self.th_lev = None

        # Each input data _must_ have u-wind, isentropic pv, and thermal tropopause,
        # but _might_ need the v-wind and air temperature to calculate PV/thermal-trop
        self.uwnd = None
        self.ipv = None
        self.trop_theta = None
        self.dyn_trop = None
        self.in_data = None
        self.strf_lat = None

    def get_data_input(self):
        """Get input data for metric calculation."""
        # First, check if we want to update data, or need to create from scratch
        # if not, then we can load existing data
        cfg = self.data_cfg

        pv_file = os.path.join(cfg['wpath'],
                               cfg['file_paths']['ipv'].format(year=self.year))
        psi_file = os.path.join(cfg['wpath'],
                                cfg['file_paths']['psi'].format(year=self.year))

        pv_update = self.config['update_pv'] or not os.path.exists(pv_file)
        psi_update = self.config['update_psi'] or not os.path.exists(psi_file)

        if pv_update or psi_update:
            self._load_data(pv_update, psi_update)

        if pv_update:
            self._calc_ipv()
            self._write_ipv()
        else:
            self._load_ipv()

        if psi_update:
            #Only need the stream function up to 100 hPa
            troposphere_only = np.where(self.lev >= 10000.)[0]
            self._calc_stream_func(troposphere_only)
            self._write_stream_func()
        else:
            self._load_stream_func()
        import pdb;pdb.set_trace()

    def check_input_range(self, year_s, year_e):
        """
        Create/check input data for a range of years.

        Parameters
        ----------
        year_s, year_e : int
            Start and end years of period, respectively

        """
        cfg = self.data_cfg
        pv_file_fmt = os.path.join(cfg['wpath'], cfg['file_paths']['ipv'])
        psi_file_fmt = os.path.join(cfg['wpath'], cfg['file_paths']['psi'])

        for year in range(year_s, year_e + 1):
            self.year = year
            self.props.log.info('CHECKING INPUT FOR {}'.format(year))
            pv_file = pv_file_fmt.format(year=self.year)
            psi_file = psi_file_fmt.format(year=self.year)
            self.props.log.info('CHECKING: {}'.format(pv_file))
            self.props.log.info('CHECKING: {}'.format(psi_file))

            pv_update = self.config['update_pv'] or not os.path.exists(pv_file)
            psi_update = self.config['update_psi'] or not os.path.exists(psi_file)

            if pv_update or psi_update:
                self._load_data(pv_update, psi_update)

            if pv_update:
                self._calc_ipv()
                self._write_ipv()

            if psi_update:
                #Only need the stream function up to 100 hPa
                troposphere_only = np.where(self.lev >= 10000.)[0]
                self._calc_stream_func(troposphere_only)
                self._write_stream_func()

    def _load_data(self, pv_update=False, psi_update=False):
        cfg = self.data_cfg
        self.in_data = {}

        data_vars = []
        if pv_update:
            data_vars.extend(['uwnd', 'vwnd', 'tair'])

        if psi_update:
            psi_vars = set(['vwnd', 'omega'])
            data_vars = list(set(data_vars).union(psi_vars))

        if cfg['ztype'] == 'theta':
            # If input data is isentropic already..need pressure on theta, not air temp
            if pv_update:
                data_vars.remove('tair')
            data_vars.append('pres')

        # This is how they're called in the configuration file, each should point to
        # how the variable is called in the actual netCDF file
        dim_vars = ['time', 'lev', 'lat', 'lon']

        # Load u/v/t; create pv file that has ipv, tpause file with tropopause lev
        first_file = True
        nc_file = None
        for var in data_vars:
            vname = cfg[var]
            if nc_file is None:
                # Format the name of the file, join it with the path, open it
                try:
                    file_name = cfg['file_paths'][var].format(year=self.year)
                except KeyError:
                    file_name = cfg['file_paths']['all'].format(year=self.year)
                self.props.log.info('OPEN: {}'.format(os.path.join(cfg['path'],
                                                                   file_name)))
                nc_file = nc.Dataset(os.path.join(cfg['path'], file_name), 'r')
            self.props.log.info("\tLOAD: {}".format(var))
            self.in_data[var] = nc_file.variables[vname][:, ...].astype(np.float16)

            if first_file:
                for dvar in dim_vars:
                    v_in_name = cfg[dvar]
                    if dvar == 'time':
                        setattr(self, dvar, nc_file.variables[v_in_name][:])
                    elif dvar == 'lev' and cfg['ztype'] == 'pres':
                        setattr(self, dvar, nc_file.variables[v_in_name][:] * cfg['pfac'])
                    else:
                        setattr(self, dvar, nc_file.variables[v_in_name][:])

                # Set time units and calendar properties
                self.time_units = nc_file.variables[cfg['time']].units
                try:
                    self.calendar = nc_file.variables[cfg['time']].calendar
                except (KeyError, AttributeError):
                    self.calendar = 'standard'

                first_file = False

            if cfg['single_var_file']:
                nc_file.close()
                nc_file = None

        if not cfg['single_var_file']:
            nc_file.close()

    def _gen_chunks(self, n_chunks=3):
        """Split data into time-period chunks if needed."""
        total_mem = psutil.virtual_memory().total
        # Data is in numpy float32, so total size is npoints * 32 / 8 in bytes
        dset_size = np.prod(self.in_data['uwnd'].shape) * 32 / 8
        ideal_chunks = int(np.floor(100 / (np.prod(self.in_data['uwnd'].shape) * 32 / 8 /
                                           psutil.virtual_memory().available * 100)))
        if ideal_chunks > n_chunks:
            n_chunks = ideal_chunks
        if (dset_size / total_mem) > 0.01:
            n_times = self.in_data['uwnd'].shape[0]
            cwidth = n_times // n_chunks
            chunks = [[ix, ix + cwidth] for ix in range(0, n_times + cwidth, cwidth)]
            # Using the above range, the last chunk generated is beyond the shape of axis0
            chunks.pop(-1)
            # Set the last element of the last chunk to None, just in case, so all data
            # gets calculated no matter how the chunks are created
            chunks[-1][-1] = None
        else:
            chunks = [(0, None)]
        return chunks

    def _calc_ipv(self):
        # Shorthand for configuration dictionary
        cfg = self.data_cfg
        if self.in_data is None:
            self._load_data()
        self.props.log.info('Starting IPV calculation')

        # calculate IPV
        if cfg['ztype'] == 'pres':
            th_shape = list(self.in_data['uwnd'].shape)
            th_shape[1] = self.props.th_levels.shape[0]

            # Pre-allocate memory for PV and Wind fields
            self.ipv = np.zeros(th_shape)
            self.uwnd = np.zeros(th_shape)
            chunks = self._gen_chunks()
            self.props.log.info('CALCULATE IPV USING {} CHUNKS'.format(len(chunks)))
            for ix_s, ix_e in chunks:
                self.ipv[ix_s:ix_e, ...], _, self.uwnd[ix_s:ix_e, ...] =\
                    utils.ipv(self.in_data['uwnd'][ix_s:ix_e, ...],
                              self.in_data['vwnd'][ix_s:ix_e, ...],
                              self.in_data['tair'][ix_s:ix_e, ...],
                              self.lev, self.lat, self.lon, self.props.th_levels)
            self.ipv *= 1e6  # Put PV in units of PVU
            self.th_lev = self.props.th_levels

        elif cfg['ztype'] == 'theta':
            self.uwnd = self.in_data['uwnd']
            self.ipv = utils.ipv_theta(self.in_data['uwnd'], self.in_data['vwnd'],
                                       self.in_data['pres'], self.lat, self.lon,
                                       self.lev)

        self.props.log.info('Finished calculating IPV')

    def _calc_stream_func(self, troposphere_only):

        data = {'omega': np.ma.mean(self.in_data['omega'][:, troposphere_only, :], axis=3),
                'vcomp': np.ma.mean(self.in_data['vwnd'][:, troposphere_only, :], axis=3),
                'pfull': self.lev[troposphere_only], 'lat': self.lat, 'time' : self.time}

        psi = cal_stream_fn(data)

        self.props.log.info('Finished calculating stream function')

        self._zero_crossing(psi, troposphere_only)

    def _zero_crossing(self, psi, troposphere_only):
        """At the 500 hPa level, find the abs. max. stream function in each hemisphere."""

        # select 500 level
        lev_500 = np.where(self.lev[troposphere_only] == 50000.)[0]
        assert len(lev_500) == 1, 'Where is 500hPa level?'
        psi_500 = np.squeeze(psi[:, lev_500, :])

        lat_arg_sort = np.argsort(self.lat)
        nh_max_elem = np.argmax(np.abs(psi_500[:, self.lat > 0]), axis=1)
        sh_max_elem = np.argmax(np.abs(psi_500[:, self.lat < 0]), axis=1)
        nh_lat = self.lat[self.lat > 0][nh_max_elem]
        sh_lat = self.lat[self.lat < 0][sh_max_elem]

        self.strf_lat = np.array([sh_lat, nh_lat]).T

        plot_test = False
        if plot_test:
            plt.plot(self.lat, psi_500[0, :])
            plt.plot([nh_lat[0], nh_lat[0]], [0, 0], c='r', marker='x')
            plt.plot([sh_lat[0], sh_lat[0]], [0, 0], c='r', marker='x')
            plt.show()

    def _write_stream_func(self, out_file=None):
        """
        Save psi (i.e stream function) data generated to a file, either netCDF4 or pickle.

        Parameters
        ----------
        out_file : string, optional
            Output file path for pickle or netCDF4 file, will contain psi data and coords

        """
        if out_file is None:
            file_name = self.data_cfg['file_paths']['psi'].format(year=self.year)
            out_file = os.path.join(self.data_cfg['wpath'], file_name)

        self.props.log.info('WRITE PSI: {}'.format(out_file))

        coord_names = ['time']
        coords = {cname: getattr(self, cname) for cname in coord_names}
        coords['lat'] = [0, 1]  # Use 'lat' as hemisphere coord to fake out data_out

        props = {'name': 'Latitude of maximum psi',
                 'descr': ' Max stream function location',
                 'units': 'degrees_north',
                 'short_name': 'psi_max',
                 'timevar': self.data_cfg['time'],
                 'latvar': 'hemis',
                 'lat_units': 'SH0_NH1',
                 'time_units': self.time_units,
                 'calendar': self.calendar}

        psi_out = dout.NCOutVar(self.strf_lat, props=props, coords=coords)

        dout.write_to_netcdf([psi_out], '{}'.format(out_file))
        self.props.log.info('Finished Writing stream function data')

    def _load_stream_func(self):
        """Load StreamFunction data from a file."""

        file_name = self.data_cfg['file_paths']['psi'].format(year=self.year)
        in_file = os.path.join(self.data_cfg['wpath'], file_name)
        lat_in = nc.Dataset(in_file, 'r')
        self.strf_lat = lat_in.variables['psi_max'][:]

        coord_names = ['time']
        for cname in coord_names:
            setattr(self, cname, lat_in.variables[self.data_cfg[cname]][:])
        if self.time_units is None:
            self.time_units = lat_in.variables[self.data_cfg['time']].units
        if self.calendar is None:
            self.calendar = lat_in.variables[self.data_cfg['time']].calendar

        lat_in.close()

    def _calc_trop(self):
        """Calculate the tropopause height using the WMO thermal definition."""
        if self.in_data is None:
            self._load_data()

        self.props.log.info('Start calculating tropopause height')
        if self.data_cfg['ztype'] == 'pres':
            if self.lev[0] < self.lev[-1]:
                self.props.log.info('INPUT DATA NOT IN sfc -> upper levels ORDER')
                v_slice = slice(None, None, -1)
            else:
                v_slice = slice(None, None, None)

            chunks = self._gen_chunks(5)
            dims = list(self.in_data['tair'].shape)
            dims.pop(1)
            trop_h_temp = np.zeros(dims)
            trop_h_pres = np.zeros(dims)

            self.props.log.info('TROPOPAUSE FOR USUNG {} CHUNKS'.format(len(chunks)))
            for ix_s, ix_e in chunks:
                trop_h_temp[ix_s:ix_e, ...], trop_h_pres[ix_s:ix_e, ...] =\
                    utils.get_tropopause_pres(self.in_data['tair']
                                              [ix_s:ix_e, v_slice, ...],
                                              self.lev[v_slice])

        elif self.data_cfg['ztype'] == 'theta':
            trop_h_temp, trop_h_pres = utils.get_tropopause_theta(self.lev,
                                                                  self.in_data['pres'])

        self.trop_theta = utils.theta(trop_h_temp, trop_h_pres)
        self.props.log.info('Finished calculating tropopause height')

    def _calc_dyn_trop(self):
        """Calculate dynamical tropopause (pv==2PVU)."""
        pv_lev = self.config['pv_value']
        pv_lev = np.array([abs(pv_lev)])

        self.props.log.info('Start calculating dynamical tropopause')
        if self.ipv is None:
            # Calculate PV
            self._load_ipv()

        # Calculate Theta on PV == 2 PVU
        _nh = [slice(None), slice(None), self.lat >= 0, slice(None)]
        _sh = [slice(None), slice(None), self.lat < 0, slice(None)]

        dyn_trop_nh = utils.vinterp(self.th_lev, self.ipv[_nh] * 1e6, pv_lev)
        dyn_trop_sh = utils.vinterp(self.th_lev, self.ipv[_sh] * 1e6, -pv_lev)
        if self.lat[0] > self.lat[-1]:
            self.dyn_trop = np.append(dyn_trop_nh, dyn_trop_sh, axis=1)
        else:
            self.dyn_trop = np.append(dyn_trop_sh, dyn_trop_nh, axis=1)

        self.props.log.info('Finished calculating dynamical tropopause')

    def _write_ipv(self, out_file=None):
        """
        Save IPV data generated to a file, either netCDF4 or pickle.

        Parameters
        ----------
        out_file : string, optional
            Output file path for pickle or netCDF4 file, will contain ipv data and coords

        """
        if out_file is None:
            file_name = self.data_cfg['file_paths']['ipv'].format(year=self.year)
            out_file = os.path.join(self.data_cfg['wpath'], file_name)

        self.props.log.info('WRITE IPV: {}'.format(out_file))

        coord_names = ['time', 'lev', 'lat', 'lon']
        coords = {cname: getattr(self, cname) for cname in coord_names}
        coords['lev'] = self.th_lev

        props = {'name': 'isentropic_potential_vorticity',
                 'descr': 'Potential vorticity on theta levels',
                 'units': 'PVU', 'short_name': 'ipv', 'levvar': self.data_cfg['lev'],
                 'latvar': self.data_cfg['lat'], 'lonvar': self.data_cfg['lon'],
                 'timevar': self.data_cfg['time'], 'time_units': self.time_units,
                 'calendar': self.calendar, 'lat_units': 'degrees_north',
                 'lon_units': 'degrees_east', 'lev_units': 'K'}

        # IPV in the file should be in 1e-6 PVU
        ipv_out = dout.NCOutVar(self.ipv * 1e-6, props=props, coords=coords)
        u_th_out = dout.NCOutVar(self.uwnd, props=dict(props), coords=coords)
        u_th_out.set_props({'name': 'zonal_wind_component',
                            'descr': 'Zonal wind on isentropic levels',
                            'units': 'm s-1', 'short_name': self.data_cfg['uwnd']})

        dout.write_to_netcdf([ipv_out, u_th_out], '{}'.format(out_file))
        self.props.log.info('Finished Writing')

    def _write_dyn_trop(self, out_file=None):
        """
        Save dynamical tropopause data generated to a file, either netCDF4 or pickle.

        Parameters
        ----------
        out_file : string, optional
            Output file path for pickle or netCDF4 file, will contain ipv data and coords

        """
        if out_file is None:
            file_name = self.data_cfg['file_paths']['dyn_trop'].format(year=self.year)
            out_file = os.path.join(self.data_cfg['wpath'], file_name)

        self.props.log.info('WRITE DYN TROP: {}'.format(out_file))

        coord_names = ['time', 'lat', 'lon']
        coords = {cname: getattr(self, cname) for cname in coord_names}
        props = {'name': 'dynamical_tropopause_theta',
                 'descr': 'Potential temperature on potential vorticity = 2PVU',
                 'units': 'K', 'short_name': 'dyntrop',
                 'latvar': self.data_cfg['lat'], 'lonvar': self.data_cfg['lon'],
                 'timevar': self.data_cfg['time'], 'time_units': self.time_units,
                 'calendar': self.calendar, 'lat_units': 'degrees_north',
                 'lon_units': 'degrees_east'}

        dyn_trop_out = dout.NCOutVar(self.dyn_trop, props=props, coords=coords)
        dout.write_to_netcdf([dyn_trop_out], '{}'.format(out_file))
        self.props.log.info('Finished Writing Dynamical Tropopause')

    def _write_trop(self, out_file=None):
        """
        Save tropopause data generated to a file, either netCDF4 or pickle.

        Parameters
        ----------
        out_file : string, optional
            Output file path for pickle or netCDF4 file, will contain ipv data and coords

        """
        if out_file is None:
            file_name = self.data_cfg['file_paths']['tpause'].format(year=self.year)
            out_file = os.path.join(self.data_cfg['wpath'], file_name)

        coord_names = ['time', 'lat', 'lon']
        coords = {cname: getattr(self, cname) for cname in coord_names}

        self.props.log.info('WRITE TROPOPAUSE: {}'.format(out_file))
        props = {'name': 'tropopause_level', 'descr': 'Tropopause potential temperature',
                 'units': 'K', 'short_name': 'trop_theta', 'time_units': self.time_units,
                 'calendar': self.calendar, 'latvar': self.data_cfg['lat'],
                 'lonvar': self.data_cfg['lon'], 'timevar': self.data_cfg['time'],
                 'lat_units': 'degrees_north', 'lon_units': 'degrees_east'}
        trop_theta_out = dout.NCOutVar(self.trop_theta, props=props, coords=coords)
        dout.write_to_netcdf([trop_theta_out], '{}'.format(out_file))
        self.props.log.info('Finished Writing')

    def _load_ipv(self):
        """Open IPV file, load into self.ipv."""
        file_name = self.data_cfg['file_paths']['ipv'].format(year=self.year)
        in_file = os.path.join(self.data_cfg['wpath'], file_name)
        ipv_in = nc.Dataset(in_file, 'r')
        self.ipv = ipv_in.variables[self.data_cfg['ipv']][:] * 1e6
        self.uwnd = ipv_in.variables[self.data_cfg['uwnd']][:]

        coord_names = ['time', 'lat', 'lon']
        for cname in coord_names:
            setattr(self, cname, ipv_in.variables[self.data_cfg[cname]][:])
        if self.time_units is None:
            self.time_units = ipv_in.variables[self.data_cfg['time']].units
        if self.calendar is None:
            self.calendar = ipv_in.variables[self.data_cfg['time']].calendar

        self.th_lev = ipv_in.variables[self.data_cfg['lev']][:]
        ipv_in.close()

    def _load_trop(self):
        """Open IPV file, load into self.ipv."""
        file_name = self.data_cfg['file_paths']['tpause'].format(year=self.year)
        in_file = os.path.join(self.data_cfg['wpath'], file_name)
        tpause_in = nc.Dataset(in_file, 'r')
        self.trop_theta = tpause_in.variables['trop_theta'][:]

        coord_names = ['time', 'lev', 'lat', 'lon']
        for cname in coord_names:
            if getattr(self, cname) is None:
                setattr(self, cname, tpause_in.variables[self.data_cfg[cname]][:])
        self.th_lev = self.lev[:]
        if self.time_units is None:
            self.time_units = tpause_in.variables[self.data_cfg['time']].units
        if self.calendar is None:
            self.calendar = tpause_in.variables[self.data_cfg['time']].calendar
        tpause_in.close()
