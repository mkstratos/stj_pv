# -*- coding: utf-8 -*-
"""Generate or load input data for STJ Metric."""
import os
import numpy as np
import pkg_resources
import datetime as dt
import xarray as xr
# Dependent code
import STJ_PV.utils as utils

__author__ = "Penelope Maher, Michael Kelleher"


def package_data(relpath, file_name):
    """Get data relative to this installed package.
    Generally used for the sample data."""
    _data_dir = pkg_resources.resource_filename('STJ_PV', relpath)
    return xr.open_dataset(os.path.join(_data_dir, file_name))


class InputData:
    """
    Contains the relevant input data and routines for an JetFindRun.

    Parameters
    ----------
    jet_find : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Object containing properties about the metric calculation
        to be performed. Used to locate correct files, and variables
        within those files.
    year : int, optional
        Year of data to load, not used when all years are in a single file

    """
    # For the default InputData class, there are no required fields
    # this should be overridden in child classes for each metric
    load_vars = []

    def __init__(self, props, date_s=None, date_e=None):
        """Initialize InputData object, using JetFindRun class."""
        self.props = props
        self.data_cfg = props.data_cfg

        if date_s is not None:
            self.year = date_s.year
        else:
            self.year = None

        self.in_data = {}
        self.out_data = {}

        self.sel = {
            self.data_cfg['time']: slice(None),
            self.data_cfg['lev']: slice(None),
            self.data_cfg['lat']: slice(None),
            self.data_cfg['lon']: slice(None),
        }

        self.chunk = {
            self.data_cfg['time']: None,
            self.data_cfg['lev']: None,
            self.data_cfg['lat']: None,
            self.data_cfg['lon']: None,
        }

        if date_s is not None or date_e is not None:
            self.sel[self.data_cfg['time']] = slice(date_s, date_e)

        self._select_setup()

    def _select_setup(self):
        """Set up selection dictionary to be passed to xarray.DataArray.sel."""
        cfg = self.data_cfg
        for cvar in ['lon', 'lev', 'lat']:
            # In the data config file, the start and end values will be
            # labeled as lon_s, lon_e (for longitude as an example)
            _start = '{}_s'.format(cvar)
            _end = '{}_e'.format(cvar)
            _skip = '{}_skp'.format(cvar)
            _slice = (
                cfg.get(_start, None),
                cfg.get(_end, None),
                cfg.get(_skip, None),
            )
            self.sel[cfg[cvar]] = slice(*_slice)
            # if _start and _end in cfg:
            #     self.sel[cfg[cvar]] = slice(cfg[_start], cfg[_end])

    def _load_data(self):
        """Load all required data."""
        for data_var in self.load_vars:
            try:
                self._load_one_file(data_var)
            except (KeyError, FileNotFoundError):
                self.props.log.info('FILE FOR %s NOT FOUND', data_var)

    def _chunk_data(self, var):
        """Re-chunk input data to ideal size."""
        self.in_data[var] = self.in_data[var].chunk(self.chunk)

    def _set_chunks(self, data, max_size=3e6, excldims=('lev', 'lat')):
        """Get ideal-ish chunks in a different way."""
        shape = data.shape
        npoints = np.prod(shape)
        excl_n = [self.data_cfg[excldim] for excldim in excldims]
        dims = [
            coord
            for coord in data.coords
            if coord in data.dims and coord not in excl_n
        ]
        chunks = {dim: data[dim].shape[0] for dim in dims}
        divis = {dim: 1 for dim in dims}
        n_iter = 0

        while npoints > max_size and n_iter < 10:
            _pts = np.prod([data[name].shape[0] for name in excl_n])
            for dim in dims:
                divis[dim] *= 2
                _pts *= chunks[dim] // divis[dim]
            npoints = _pts
            n_iter += 1
        self.props.log.info(f'  Chunking took {n_iter} iterations')
        chunks_out = {dim: chunks[dim] // divis[dim] for dim in dims}
        for exname in excl_n:
            chunks_out[exname] = data[exname].shape[0]

        for dim in chunks_out:
            # Quick sanity check to make sure we don't divide by 0
            if chunks_out[dim] == 0:
                chunks_out[dim] = 1

            self.props.log.info(f'    - {dim}: {chunks_out[dim]}')
        self.props.log.info(f'  Chunk size: {npoints}')

        self.chunk = chunks_out

    def _load_one_file(self, var, file_var=None):
        """Load a single netCDF file as an xarray.Dataset."""
        cfg = self.data_cfg
        vname = cfg[var]

        # Use this to set the file variable name (look for uwnd in ipv file)
        if file_var is None:
            file_var = var

        # Format the name of the file, join it with the path, open it
        try:
            file_name = cfg['file_paths'][file_var].format(year=self.year)
        except KeyError:
            file_name = cfg['file_paths']['all'].format(year=self.year)

        self.props.log.info(
            'OPEN: {}'.format(os.path.join(cfg['path'], file_name))
        )
        try:
            nc_file = xr.open_dataset(os.path.join(cfg['path'], file_name))
        except FileNotFoundError:
            nc_file = package_data(cfg['path'], file_name)

        self.in_data[var] = nc_file[vname].sel(**self.sel)
        _fails = 0
        while self.in_data[var][cfg['time']].shape[0] == 0 and _fails < 15:
            # Update the time slice so that it covers potential mis-match
            # between how days are requested and how they're stored in the
            # netCDF file (e.g. ask for 2013-07-14 00:00, but the netCDF
            # file has 2013-07-14 09:00)
            _day = dt.timedelta(hours=23)
            self.sel[cfg['time']] = slice(
                self.sel[cfg['time']].start, self.sel[cfg['time']].stop + _day
            )
            self.in_data[var] = nc_file[vname].sel(**self.sel)
            self.props.log.info(
                'UPDATING TIME SLICE BY 1 DAY %s',
                (self.sel[cfg['time']].stop.strftime('%Y-%m-%d %HZ')),
            )

            # Iterate, but don't get stuck here
            _fails += 1

        if all([self.chunk[var] is None for var in self.chunk]):
            self._set_chunks(self.in_data[var])

        self._chunk_data(var)

    def get_data(self):
        """Get a single xarray.Dataset of required components for metric."""
        data = xr.Dataset(
            self.out_data, attrs={'cfg': self.data_cfg, 'year': self.year}
        )
        return data

    def write_data(self, out_file=None):
        """Write netCDF file of the out_data."""
        if out_file is None:
            file_name = self.data_cfg['file_paths']['ipv'].format(
                year=self.year
            )
            out_file = os.path.join(self.data_cfg['wpath'], file_name)

        if not os.access(out_file, os.W_OK):
            write_dir = pkg_resources.resource_filename(
                'STJ_PV', self.data_cfg['wpath']
            )
            out_file = os.path.join(write_dir, file_name)

        self.props.log.info('Begin writing to %s', out_file)
        self.get_data().to_netcdf(out_file)
        self.props.log.info('Finished writing to %s', out_file)


class InputDataSTJPV(InputData):
    """
    Contains the relevant input data and routines for an STJPV jet find.

    Parameters
    ----------
    jet_find : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Object containing properties about the metric calculation
        to be performed. Used to locate correct files, and variables
        within those files.
    year : int, optional
        Year of data to load, not used when all years are in a single file

    """

    load_vars = ['uwnd', 'vwnd', 'tair', 'epv', 'ipv']

    def __init__(self, props, date_s=None, date_e=None):
        """Initialize InputData object, using JetFindRun class."""
        super(InputDataSTJPV, self).__init__(props, date_s, date_e)

        # Each STJPV input data _must_ have u-wind and isentropic pv
        # but _might_ also need the v-wind and air temperature to
        # calculate isentropic pv
        self.out_data = {'uwnd': None, 'ipv': None}
        self.th_lev = None

    def _find_pv_update(self):
        """Determine if PV needs to be computed/re-computed."""
        pv_file_name = self.data_cfg['file_paths']['ipv'].format(
            year=self.year
        )
        pv_file = os.path.join(self.data_cfg['wpath'], pv_file_name)
        return self.props.config['update_pv'] or not os.path.exists(pv_file)

    def _calc_ipv(self):
        # Shorthand for configuration dictionary
        cfg = self.data_cfg
        dimvars = {cvar: cfg[cvar] for cvar in ['time', 'lev', 'lat', 'lon']}
        if not self.in_data:
            self._load_data()
        self.props.log.info('Starting IPV calculation')
        # calculate IPV
        if cfg['ztype'] == 'pres':
            if 'epv' not in self.in_data:
                self.props.log.info('USING U, V, T TO COMPUTE IPV')
                ipv, _, uwnd = utils.xripv(
                    self.in_data['uwnd'],
                    self.in_data['vwnd'],
                    self.in_data['tair'],
                    dimvars=dimvars,
                    th_levels=self.props.th_levels,
                )

            else:
                self.props.log.info('USING ISOBARIC PV TO COMPUTE IPV')
                thta = utils.xrtheta(self.in_data['tair'], pvar=cfg['lev'])
                ipv = utils.xrvinterp(
                    self.in_data['epv'],
                    thta,
                    self.props.th_levels,
                    levname=cfg['lev'],
                    newlevname=cfg['lev'],
                )

                uwnd = utils.xrvinterp(
                    self.in_data['uwnd'],
                    thta,
                    self.props.th_levels,
                    levname=cfg['lev'],
                    newlevname=cfg['lev'],
                )

            self.out_data['ipv'] = ipv
            self.out_data['uwnd'] = uwnd

            self.th_lev = self.props.th_levels

        elif cfg['ztype'] == 'theta':
            ipv = utils.xripv_theta(
                self.in_data['uwnd'],
                self.in_data['vwnd'],
                self.in_data['pres'],
                dimvars,
            )
            self.out_data['ipv'] = ipv
            self.out_data['uwnd'] = self.in_data['uwnd']

        ipv_attrs = {
            'units': '10^-6 PVU',
            'standard_name': 'isentropic_potential_vorticity',
            'descr': 'Potential vorticity on isentropic levels',
        }
        uwnd_attrs = {
            'units': 'm s-1',
            'standard_name': 'zonal_wind_component',
            'descr': 'Zonal wind on isentropic levels',
        }

        self.out_data['ipv'] = self.out_data['ipv'].assign_attrs(ipv_attrs)
        self.out_data['uwnd'] = self.out_data['uwnd'].assign_attrs(uwnd_attrs)
        self.props.log.info('Finished calculating IPV')

    def _load_ipv(self):
        """Open IPV and Isentropic U-wind file(s), load into self.out_data."""
        file_name = self.data_cfg['file_paths']['ipv'].format(year=self.year)
        in_file = os.path.join(self.data_cfg['wpath'], file_name)
        self.props.log.info("LOAD IPV FROM FILE: {}".format(in_file))
        self._load_one_file('ipv')
        try:
            # Check for uwind in the IPV file first
            self._load_one_file('uwnd', file_var='ipv')
        except KeyError:
            # But fall back on the uwind file
            self._load_one_file('uwnd')

        self.out_data = self.in_data
        self.th_lev = self.in_data['ipv'][self.data_cfg['lev']]

    def _write_ipv(self):
        """Write generated IPV data to file."""
        pv_file_name = self.data_cfg['file_paths']['ipv'].format(
            year=self.year
        )
        pv_file = os.path.join(self.data_cfg['wpath'], pv_file_name)
        self.props.log.info('WRITING PV FILE %s', pv_file)
        encoding = {'zlib': True, 'complevel': 9}

        dsout = xr.Dataset(self.out_data)
        dsout[self.data_cfg['lev']] = dsout[self.data_cfg['lev']].assign_attrs(
            {'units': 'K', 'standard_name': 'potential_temperature'}
        )
        dsout.encoding = dict((var, encoding) for var in dsout.data_vars)
        dsout.to_netcdf(pv_file, encoding=dsout.encoding)
        self.props.log.info('DONE WRITING PV FILE')

    def get_data(self):
        """Load and compute required data, return xarray.Dataset."""
        if 'force_write' in self.props.config:
            force_write = self.props.config['force_write']
        else:
            force_write = False

        if self._find_pv_update():
            self._calc_ipv()
            if self.sel[self.data_cfg['time']] == slice(None) or force_write:
                self._write_ipv()

        else:
            self._load_ipv()

        if self.th_lev[0] > self.th_lev[-1]:
            for data_var in ['uwnd', 'ipv']:
                self.out_data[data_var] = self.out_data[data_var][:, ::-1]
            self.th_lev = self.th_lev[::-1]

        return xr.Dataset(
            self.out_data, attrs={'cfg': self.data_cfg, 'year': self.year}
        )


class InputDataUWind(InputData):
    """
    Contains the relevant input data and routines for an STJPV jet find.

    Parameters
    ----------
    jet_find : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Object containing properties about the metric calculation
        to be performed. Used to locate correct files, and variables
        within those files.
    year : int, optional
        Year of data to load, not used when all years are in a single file

    """


    def __init__(self, props, date_s=None, date_e=None, vwnd=False):
        """Initialize InputData object, using JetFindRun class."""
        self.load_vars = ['uwnd']
        if vwnd:
            self.load_vars.append("vwnd")
        super(InputDataUWind, self).__init__(props, date_s, date_e)

        # Each UWind input data _must_ have u-wind
        # but _might_ also need the pressure calculate isobaric uwind
        self.out_data = {}
        for var_name in self.load_vars:
            self.out_data[var_name] = None

    def _calc_interp(self, var_name):
        lev = self.props.p_levels
        data_interp = utils.xrvinterp(
            self.in_data[var_name],
            self.in_data['pres'],
            lev,
            levname=self.data_cfg['lev'],
            newlevname='pres',
        )

        self.out_data[var_name] = data_interp

    def get_data(self):
        """Load and compute (if needed) U-Wind on selected pressure level."""
        if self.data_cfg == 'theta':
            self.load_vars.append('pres')
            self._load_data()
            for var_name in self.out_data:
                self._calc_interp(var_name)
        else:
            self._load_data()
            self.out_data = self.in_data

        return xr.Dataset(
            self.out_data, attrs={'cfg': self.data_cfg, 'year': self.year}
        )
