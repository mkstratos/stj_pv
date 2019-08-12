# -*- coding: utf-8 -*-
"""STJ Metric: Calculate the position of the subtropical jet in both hemispheres."""
import subprocess
import numpy as np
import numpy.polynomial as poly
from scipy import signal as sig
from scipy.signal import argrelextrema

import utils
import STJ_PV.data_out as dio

from netCDF4 import num2date, date2num
import pandas as pd
import xarray as xr

import pdb

try:
    from eddy_terms import Kinetic_Eddy_Energies
except ModuleNotFoundError:
    print('Eddy Terms Function not available, STJKangPolvani not available')

try:
    GIT_ID = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
except subprocess.CalledProcessError as err:
    GIT_ID = 'NONE'


class STJMetric(object):
    """Generic Class containing Sub Tropical Jet metric methods and attributes."""

    def __init__(self, name=None, data=None, props=None):
        """
        Initialize a subtropical jet metric.

        Parameters
        ----------
        name : string
            Name of this type of method
        data : InputData
            Object containing required input data
        props : JetFindRun
            Properties about the current jet finding attempt, including log file

        """
        self.name = name
        self.data = data
        self.props = props.config
        self.log = props.log
        self.jet_lat = None
        self.jet_theta = None
        self.jet_intens = None
        self.time = None
        self.hemis = None
        self.debug_data = {}
        self.plot_idx = 0

    def save_jet(self):
        """Save jet position to file."""
        # Create output variables
        props_lat = {'name': 'jet_latitude', 'descr': 'Latitude of subtropical jet',
                     'units': 'degrees_north', 'short_name': 'lat_sh', 'timevar': 'time',
                     'calendar': self.data.calendar, 'time_units': self.data.time_units}
        coords = {'time': self.time}

        self.log.info("CREATE OUTPUT VARIABLES")

        # Latitude
        if self.props['zonal_opt'].lower() != 'mean':
            props_lat['lonvar'] = 'lon'
            props_lat['lon_units'] = 'degrees_east'
            coords['lon'] = self.data.lon

        props_lat_nh = dict(props_lat)
        props_lat_nh['short_name'] = 'lat_nh'

        lat_sh_out = dio.NCOutVar(self.jet_lat[0, ...], coords=coords, props=props_lat)
        lat_nh_out = dio.NCOutVar(self.jet_lat[1, ...], coords=coords, props=props_lat_nh)

        # Intensity
        props_int = dict(props_lat)
        props_int['name'] = 'jet_intensity'
        props_int['descr'] = 'Intensity of subtropical jet'
        props_int['units'] = 'm s-1'
        props_int['short_name'] = 'intens_sh'

        props_int_nh = dict(props_int)
        props_int_nh['short_name'] = 'intens_nh'

        intens_sh_out = dio.NCOutVar(self.jet_intens[0, ...], coords=coords,
                                     props=props_int)
        intens_nh_out = dio.NCOutVar(self.jet_intens[1, ...], coords=coords,
                                     props=props_int_nh)

        # Theta (only if Theta is valid)
        if self.jet_theta is not None:
            props_th = dict(props_lat)
            props_th['name'] = 'jet_theta'
            props_th['descr'] = 'Theta level of subtropical jet'
            props_th['units'] = 'K'
            props_th['short_name'] = 'theta_sh'
            theta_sh_out = dio.NCOutVar(self.jet_theta[0, ...], coords=coords,
                                        props=props_th)
            props_th_nh = dict(props_th)
            props_th_nh['short_name'] = 'theta_nh'
            theta_nh_out = dio.NCOutVar(self.jet_theta[1, ...], coords=coords,
                                        props=props_th_nh)

        self.log.info("WRITE TO {output_file}".format(**self.props))

        # Write jet & theta positions to file
        out_vars = [lat_sh_out, lat_nh_out, intens_sh_out, intens_nh_out]
        if self.jet_theta is not None:
            out_vars.extend([theta_sh_out, theta_nh_out])

        file_attrs = [('commit-id', GIT_ID), ('run_props', self.props)]

        dio.write_to_netcdf(out_vars, self.props['output_file'] + '.nc', file_attrs)

    def append(self, other):
        """Append another metric's latitude and theta positon to this one."""
        self.jet_lat = np.append(self.jet_lat, other.jet_lat, axis=1)
        self.jet_intens = np.append(self.jet_intens, other.jet_intens, axis=1)
        if self.jet_theta is not None:
            self.jet_theta = np.append(self.jet_theta, other.jet_theta, axis=1)

        # Double check the units!
        if self.data.time_units == other.data.time_units:
            self.time = np.append(self.time, other.time, axis=0)
        else:
            # Convert other's time to our time
            _other_dates = num2date(other.time, other.data.time_units)
            _other_time = date2num(_other_dates, self.data.time_units)
            self.time = np.append(self.time, _other_time, axis=0)


class STJPV(STJMetric):
    """
    Subtropical jet position metric using dynamic tropopause on isentropic levels.

    Parameters
    ----------
    props : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Class containing properties about the current search for the STJ
    data : :py:meth:`~STJ_PV.input_data.InputData`
        Input data class containing a year (or more) of required data

    """

    def __init__(self, props, data):
        """Initialise Metric using PV Gradient Method."""
        name = 'PVGrad'
        super(STJPV, self).__init__(name=name, props=props, data=data)

        if np.max(np.abs(self.data.ipv)) < 1.0:
            self.data.ipv *= 1e6    # Put PV into units of PVU from 1e-6 PVU

        # Some config options should be properties for ease of access
        self.pv_lev = self.props['pv_value']
        self.fit_deg = self.props['fit_deg']
        self.min_lat = self.props['min_lat']

        if self.props['poly'].lower() in ['cheby', 'cby', 'cheb', 'chebyshev']:
            self.pfit = poly.chebyshev.chebfit
            self.pder = poly.chebyshev.chebder
            self.peval = poly.chebyshev.chebval

        elif self.props['poly'].lower() in ['leg', 'legen', 'legendre']:
            self.pfit = poly.legendre.legfit
            self.pder = poly.legendre.legder
            self.peval = poly.legendre.legval

        elif self.props['poly'].lower() in ['poly', 'polynomial']:
            self.pfit = poly.polynomial.polyfit
            self.pder = poly.polynomial.polyder
            self.peval = poly.polynomial.polyval

        # Initialise latitude & theta output arrays with correct shape
        dims = self.data.ipv.shape
        if self.props['zonal_opt'].lower() == 'mean':
            self.jet_lat = np.zeros([2, dims[0]])
            self.jet_theta = np.zeros([2, dims[0]])
            self.jet_intens = np.zeros([2, dims[0]])
        else:
            self.jet_lat = np.zeros([2, dims[0], dims[-1]])
            self.jet_theta = np.zeros([2, dims[0], dims[-1]])
            self.jet_intens = np.zeros([2, dims[0], dims[-1]])

        self.time = self.data.time[:]
        self.tix = None
        self.xix = None

    def _poly_deriv(self, lat, data, y_s=None, y_e=None, deriv=1):
        """
        Calculate the `deriv`^th derivative of a one-dimensional array w.r.t. latitude.

        Parameters
        ----------
        data : array_like
            1D array of data, same shape as `self.data.lat`
        y_s, y_e : integers, optional
            Start and end indices of subset, default is None
        deriv : integer, optional
            Number of derivatives of `data` to take

        Returns
        -------
        poly_der : array_like
            1D array of 1st derivative of data w.r.t. latitude between indices y_s and y_e

        """
        # Determine where data is valid...Intel's Lin Alg routines fail when trying to do
        # a least squares fit on array with np.nan, use where it's valid to do the fit
        valid = np.isfinite(data[y_s:y_e])
        poly_fit = self.pfit(lat[y_s:y_e][valid], data[y_s:y_e][valid], self.fit_deg)
        poly_der = self.peval(lat[y_s:y_e], self.pder(poly_fit, deriv))

        return poly_der, (poly_fit, lat[y_s:y_e][valid])

    def set_hemis(self, shemis):
        """
        Select hemisphere data.

        This function sets `self.hemis` to be an length N list of slices such that only
        the desired hemisphere is selected with N-D data (e.g. uwind and ipv) along all
        other axes. It also returns the latitude for the selected hemisphere, an index
        to select the hemisphere in output arrays, and the extrema function to find
        min/max of PV derivative in a particular hemisphere.

        Parameters
        ----------
        shemis : boolean
            If true - use southern hemisphere data, if false, use NH data

        Returns
        -------
        lat : array_like
            Latitude array from selected hemisphere
        hidx : int
            Hemisphere index 0 for SH, 1 for NH
        extrema : function
            Function used to identify extrema in meridional PV gradient, either
            :func:`scipy.signal.argrelmax` if SH, or :func:`scipy.signal.argrelmin`
            for NH

        """

        # Find axis
        lat_axis = self.data.ipv.shape.index(self.data.lat.shape[0])

        if self.data.ipv.shape.count(self.data.lat.shape[0]) > 1:
            # Log a message about which matching dimension used since this
            # could be time or lev or lon if ntimes, nlevs or nlons == nlats
            self.log.info('ASSUMING LAT DIM IS: {} ({})'.format(lat_axis,
                                                                self.data.ipv.shape))

        self.hemis = [slice(None)] * self.data.ipv.ndim

        if shemis:
            self.hemis[lat_axis] = self.data.lat < 0
            lat = self.data.lat[self.data.lat < 0]
            hidx = 0
            # Link `extrema` function to argrelmax for SH
            extrema = sig.argrelmax
        else:
            self.hemis[lat_axis] = self.data.lat > 0
            lat = self.data.lat[self.data.lat > 0]
            hidx = 1
            # Link `extrema` function to argrelmin for NH
            extrema = sig.argrelmin

        return lat, hidx, extrema

    def isolate_pv(self, pv_lev, theta_bnds=None):
        """
        Get the potential temperature, zonal wind and zonal wind shear for a PV level.

        Parameters
        ----------
        pv_lev : float
            PV value (for a particular hemisphere, >0 for NH, <0 for SH) on which to
            interpolate potential temperature and wind
        theta_bnds : tuple, optional
            Start and end theta levels to use for interpolation. Default is None,
            if None, use all theta levels, otherwise restrict so
            theta_bnds[0] <= theta <= theta_bnds[1]

        Returns
        -------
        theta_xpv : array_like
            N-1 dimensional array (where `self.data.ipv` is N-D) of potential temperature
            on `pv_lev` PVU
        uwnd_xpv : array_like
            N-1 dimensional array (where `self.data.uwnd` is N-D) of zonal wind
            on `pv_lev` PVU
        ushear : array_like
            Wind shear between uwnd_xpv and "surface", meaning the lowest valid level

        """
        # Temporary? Need to use for JRA data, it doesn't interpolate correctly with
        # all the data...?
        if theta_bnds is None:
            th_slice = slice(None)
        else:
            assert theta_bnds[0] < theta_bnds[1], 'Start level not strictly less than end'
            th_slice = np.logical_and(self.data.th_lev >= theta_bnds[0],
                                      self.data.th_lev <= theta_bnds[1])
        theta_xpv = utils.vinterp(self.data.th_lev[th_slice],
                                  self.data.ipv[self.hemis][:, th_slice, ...], pv_lev)

        uwnd_xpv = utils.vinterp(self.data.uwnd[self.hemis][:, th_slice, ...],
                                 self.data.ipv[self.hemis][:, th_slice, ...], pv_lev)

        ushear = self._get_max_shear(uwnd_xpv)
        return theta_xpv, uwnd_xpv, ushear

    def find_jet(self, shemis=True):
        """
        Find the subtropical jet using input parameters.

        Parameters
        ----------
        shemis : logical, optional
            If True, find jet position in Southern Hemisphere, if False, find N.H. jet

        """
        if shemis and self.pv_lev < 0 or not shemis and self.pv_lev > 0:
            pv_lev = np.array([self.pv_lev])
        else:
            pv_lev = -1 * np.array([self.pv_lev])

        lat, hidx, extrema = self.set_hemis(shemis)

        # Get theta on PV==pv_level
        if 'theta_s' in self.data.data_cfg and 'theta_e' in self.data.data_cfg:
            theta_bnds = (self.data.data_cfg['theta_s'], self.data.data_cfg['theta_e'])
        else:
            theta_bnds = None

        theta_xpv, uwnd_xpv, ushear = self.isolate_pv(pv_lev, theta_bnds=theta_bnds)
        dims = theta_xpv.shape

        self.log.info('COMPUTING JET POSITION FOR %d TIMES HEMIS: %d', dims[0], hidx)
        for tix in range(dims[0]):
            if tix % 50 == 0 and dims[0] > 50:
                self.log.info('COMPUTING JET POSITION FOR %d', tix)
            self.tix = tix
            jet_loc = np.zeros(dims[-1], dtype=int)
            for xix in range(dims[-1]):
                self.xix = xix
                try:
                    jet_loc[xix] = self.find_single_jet(theta_xpv[tix, :, xix],
                                                        lat, ushear[tix, :, xix],
                                                        extrema)
                except TypeError as err:
                    # This can happen on fitting the polynomial:
                    # `raise TypeError("expected non-empty vector for x")`
                    # If that's the error we get, just set the position to 0,
                    # which is later masked, otherwise raise the error
                    if 'non-empty' in err.args[0]:
                        jet_loc[xix] = 0
                    else:
                        raise

                if not self.props['zonal_opt'].lower() == 'mean':
                    self.jet_lat[hidx, tix, xix] = lat[jet_loc[xix]]
                    self.jet_theta[hidx, tix, xix] = theta_xpv[tix, jet_loc[xix], xix]
                    self.jet_intens[hidx, tix, xix] = uwnd_xpv[tix, jet_loc[xix], xix]

            if self.props['zonal_opt'].lower() in ['mean', 'median']:
                if self.props['zonal_opt'].lower() == 'mean':
                    reduce_fcn = np.ma.mean
                else:
                    reduce_fcn = np.ma.median

                jet_lat = np.ma.masked_where(jet_loc == 0, lat[jet_loc.astype(int)])
                self.jet_lat[hidx, tix] = reduce_fcn(np.ma.masked_invalid(jet_lat))

                # First take the zonal median of Theta on dyn. tropopause
                jet_theta = reduce_fcn(theta_xpv[tix, :, :], axis=-1)
                # Mask wherever jet_loc is undefined, jet_loc is a func. of longitude here
                jet_theta = np.ma.masked_where(jet_loc == 0,
                                               jet_theta[jet_loc.astype(int)])
                # Then save the zonal median of this to the correct position in output
                self.jet_theta[hidx, tix] = reduce_fcn(np.ma.masked_invalid(jet_theta))

                # Now do the same for jet intensity
                jet_intens = reduce_fcn(uwnd_xpv[tix, :, :], axis=-1)
                jet_intens = np.ma.masked_where(jet_loc == 0,
                                                jet_intens[jet_loc.astype(int)])
                self.jet_intens[hidx, tix] = reduce_fcn(np.ma.masked_invalid(jet_intens))

    def _get_max_shear(self, uwnd_xpv):
        """Get maximum wind-shear between surface and PV surface."""
        # Our zonal wind data is on isentropic levels. Lower levels are bound to be below
        # the surface in some places, so we need to use the lowest valid wind level as
        # the surface, so do some magic to make that happen.

        # Put our vertical dimension as the last dim
        u_hem = np.swapaxes(self.data.uwnd[self.hemis], 1, -1)
        # Get the first valid index on the vertical dim
        valid_idx = np.isfinite(u_hem).argmax(axis=-1)
        # Get the shape of our swapped data
        shape = list(u_hem.shape)
        n_z = shape.pop(-1)
        uwnd_sfc = u_hem.reshape(-1, n_z)[np.arange(np.prod(shape)),
                                          valid_idx.ravel()].reshape(shape)

        uwnd_sfc = np.swapaxes(uwnd_sfc, 1, -1)

        return uwnd_xpv - uwnd_sfc

    def find_single_jet(self, theta_xpv, lat, ushear, extrema, debug=False):
        """
        Find jet location for a 1D array of theta on latitude.

        Parameters
        ----------
        theta_xpv : array_like
            Theta on PV level as a function of latitude
        lat : array_like
            1D array of latitude same shape as theta_xpv from :py:meth:`~isolate_pv`
        ushear : array_like
            1D array along latitude axis of maximum surface - troposphere u-wind shear
        debug : boolean
            If True, returns debugging information about how jet position is found,
            if False (default) returns only jet location

        Returns
        -------
        jet_loc : int
            If debug is False, Index of jet location on latitude axis
        jet_loc, jet_loc_all, dtheta, theta_fit, lat, y_s, y_e  : tuple
            If debug is True, return lots of stuff
            TODO: document this better!!

        """
        # Restrict interpolation domain to a "reasonable" subset using a minimum latitude
        y_s = np.abs(np.abs(lat) - self.props['min_lat']).argmin()
        y_e = np.abs(np.abs(lat) - self.props['max_lat']).argmin()

        # If latitude is in decreasing order, switch start & end
        # This makes sure we're selecting the latitude nearest the equator
        if abs(lat[0]) > abs(lat[-1]):
            y_s, y_e = y_e, y_s
        # Find derivative of dynamical tropopause
        dtheta, theta_fit = self._poly_deriv(lat, theta_xpv, y_s=y_s, y_e=y_e)

        jet_loc_all = extrema(dtheta)[0].astype(int)
        if y_s is not None:
            # If beginning of array is cut off rather than end, add cut-off to adjust
            jet_loc_all += y_s
        select = self.select_jet(jet_loc_all, ushear)

        # Eventually this moves somewhere else to do diagnostic plots
        # if self.plot_idx <= 30:
        #     if np.min(lat) < 0 and self.xix < 2:
        #         self._debug_plot(lat, theta_xpv, theta_fit, dtheta, jet_loc_all,
        #                          y_s, y_e, select)
        if debug:
            output = select, jet_loc_all, dtheta, theta_fit, lat, y_s, y_e
        else:
            output = select

        return output

    def select_jet(self, locs, ushear):
        """
        Select correct jet latitude given list of possible jet locations.

        Parameters
        ----------
        locs : list
            List of indicies of jet locations
        ushear : array_like
            1D array along latitude axis of maximum surface - troposphere u-wind shear

        Returns
        -------
        jet_loc : int
            Index of the jet location. Between [`0`, `lat.shape[0] - 1`]

        Notes
        -----
        * If the list of locations is empty, return ``0`` as the location, this is
          interpreted by :py:meth:`~find_jet` as missing.

        * If the list of locations is exactly one return that location.

        * Otherwise use the location with maximum zonal wind shear between lowest
          provided level and the dynamical tropopause.

        """
        if len(locs) == 0:
            # A jet has not been identified at this time/location, set the position
            # to zero so it can be masked out when the zonal median is performed
            jet_loc = 0

        elif len(locs) == 1:
            # This essentially converts a list of length 1 to a single int
            jet_loc = locs[0]

        elif len(locs) > 1:
            # The jet location, if multiple peaks are identified, should be the one
            # with maximum wind shear between the jet level and the surface
            ushear_max = np.argmax(ushear[locs])
            jet_loc = locs[ushear_max]

        return jet_loc

class STJDavisBirner(STJMetric):
    """
    Subtropical jet position metric using the method of Davis and Birner 2016.
       "Climate Model Biases in the Width of the Tropical Belt
        Parameters".
    The logic for the method is to subtract the surface wind and then find the
    max in the upper level wind.
    ----------
    props : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Class containing properties about the current search for the STJ
    data : :py:meth:`~STJ_PV.input_data.InputData`
        Input data class containing a year (or more) of required data

    """

    # https://github.com/TropD/pytropd/blob/master/pytropd/metrics.py

    def __init__(self, props, data):
        """Initialise Metric using Davis and Birner 2016 method."""
        name = 'DavisBirner'
        super(STJDavisBirner, self).__init__(name=name, props=props, data=data)

        # Some config options should be properties for ease of access
        self.upper_p_level = self.props['upper_p_level']
        self.lower_p_level = self.props['lower_p_level']
        self.surf_p_level = self.props['surface_p_level']

        # Initialise latitude & theta output arrays with correct shape
        dims = self.data.uwnd.shape

        self.jet_lat = np.zeros([2, dims[0]])
        self.jet_intens = np.zeros([2, dims[0]])

        self.time = self.data.time[:]
        self.tix = None
        self.xix = None

    def prep_data(self):
        wh_surf = np.where(self.data.lev == self.surf_p_level)[0]

        self.surface_wind = self.data.uwnd[:, wh_surf, ...].squeeze()

    def find_jet(self, shemis=True):
        """
        Find the subtropical jet using method from Davis and Birner (2016).

        doi:10.1175/JCLI-D-15-0336.1

        Parameters
        ----------
        shemis : logical, optional
            If True, find jet position in Southern Hemisphere, if False, find N.H. jet

        """
        # Find axis
        lat_axis = self.data.uwnd.shape.index(self.data.lat.shape[0])

        if self.data.uwnd.shape.count(self.data.lat.shape[0]) > 1:
            # Log a message about which matching dimension used since this
            # could be time or lev or lon if ntimes, nlevs or nlons == nlats
            self.log.info(
                'ASSUMING LAT DIM IS: {} ({})'.format(lat_axis, self.data.uwnd.shape)
            )

        self.hemis = [slice(None)] * self.data.uwnd.ndim

        if shemis:
            self.hemis[lat_axis] = self.data.lat < 0
            lat = self.data.lat[self.data.lat < 0]
            hidx = 0
        else:
            self.hemis[lat_axis] = self.data.lat > 0
            lat = self.data.lat[self.data.lat > 0]
            hidx = 1

        # Isolate wind between 400-100 hpa
        wh_upper = np.where(self.data.lev == self.upper_p_level)[0]
        wh_lower = np.where(self.data.lev == self.lower_p_level)[0]

        if wh_upper > wh_lower:
            uwnd_p = self.data.uwnd[self.hemis][:, wh_lower[0] : wh_upper[0] + 1, ...]
            pres = self.data.lev[wh_lower[0] : wh_upper[0] + 1]
        else:
            uwnd_p = self.data.uwnd[self.hemis][:, wh_upper[0] : wh_lower[0] + 1, ...]
            pres = self.data.lev[wh_upper[0] : wh_lower[0] + 1]

        # Subtract the surface wind from the wind in 400-100 layer
        wh_surf = np.where(self.data.lev == self.surf_p_level)[0]
        for pidx in range(uwnd_p.shape[1]):
            uwnd_p[:, pidx, ...] = np.squeeze(
                uwnd_p[:, pidx, ...] - self.data.uwnd[self.hemis][:, wh_surf[0], ...]
            )

        dims = uwnd_p.shape

        self.log.info('COMPUTING JET POSITION FOR %d TIMES HEMIS: %d', dims[0], hidx)
        for self.tix in range(dims[0]):
            if self.tix % 50 == 0 and dims[0] > 50:
                self.log.info('COMPUTING JET POSITION FOR %d', self.tix)

            # find the maximum zonal wind in the layer
            uzonal = uwnd_p[self.tix, :, :].mean(axis=-1)
            stjp, stji = self.find_max_wind_surface(uzonal, pres, lat, shemis)

            self.jet_lat[hidx, self.tix] = stjp
            self.jet_intens[hidx, self.tix] = stji

            # None of this has been interpolated. Should it be?

    def find_max_wind_surface(self, uzonal, pres, lat, shemis):
        """
        Find most equatorward maximum wind on column maximum wind surface.

        Parameters
        ----------
        uzonal : array_like
            Zonal mean zonal wind
        pres : array_like
            Pressure coordinate array
        lat : array_like
            Latitude coordinate array
        shemis : bool
            Flag to indicate hemisphere, true for SH

        """
        # find the max wind surface and then keep the most equatorward latitude
        if shemis:
            # I had to increase this to remove false hits
            lat_valid_idx = np.where(lat < -10)[0]
        else:
            lat_valid_idx = np.where(lat > 10)[0]

        max_wind_surface_idx = np.argmax(uzonal[:, lat_valid_idx], axis=0)
        max_wind_surface = np.max(uzonal[:, lat_valid_idx], axis=0)

        # for the given maximum wind surface, find local
        # maxima and then keep most equatorward.
        turning_points = argrelextrema(max_wind_surface, np.greater_equal)[0]

        if shemis:
            lat_idx = np.min(turning_points)
            if lat[lat_valid_idx][lat_idx] >= -15:
                pdb.set_trace()

        else:
            lat_idx = np.max(turning_points)

        _lat = lat[lat_valid_idx]

        if lat_idx != 0 and lat_idx != _lat.shape[0]:
            # If the selected index is away from the boundaries, interpolate using
            # a quadratic to find the "real" maximum location
            nearby = slice(lat_idx - 1, lat_idx + 2)
            # Cast lat and max_wind_surface as arrays of float32
            # because linalg can't handle float16
            pfit = np.polyfit(
                _lat[nearby].astype(np.float32),
                max_wind_surface[nearby].astype(np.float32),
                deg=2,
            )
            # Take first derivative of the quadratic, solve for maximum
            # to get the jet latitude
            stj_lat = -pfit[1] / (2 * pfit[0])

            # Evaluate 2nd degree polynomial at the maximum to get intensity
            stj_intens = np.polyval(pfit, stj_lat)
        else:
            stj_lat = lat[lat_valid_idx][lat_idx]
            stj_intens = max_wind_surface[lat_idx]

        test_plot = True
        if test_plot:
            # run the code for a given year and see the location
            print("jet intensity is: ", stj_intens)
            import matplotlib.pyplot as plt

            plot_line = False

            if shemis:
                hem = 'SH'
                if stj_lat < -30.0:
                    plot_line = True
            else:
                hem = 'NH'
                if self.tix == 0:
                    plt.close()
                if stj_lat > 30.0:
                    plot_line = True

            if plot_line:
                plt.plot(lat[lat_valid_idx], max_wind_surface)
                plt.plot(stj_lat, max_wind_surface[lat_idx], 'x')

            if self.tix == 11:
                plt.title(hem)
                filename = ('test_davis_{0}.png').format(hem)
                plt.savefig(filename)

            # plt.show()

        return stj_lat, stj_intens


class STJMaxWind(STJMetric):
    """
    Subtropical jet position metric: maximum zonal mean zonal wind on a pressure level.

    Parameters
    ----------
    props : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Class containing properties about the current search for the STJ
    data : :py:meth:`~STJ_PV.input_data.InputData`
        Input data class containing a year (or more) of required data

    """

    def __init__(self, props, data):
        """Initialise Metric using PV Gradient Method."""
        name = 'UMax'
        super(STJMaxWind, self).__init__(name=name, props=props, data=data)

        # Some config options should be properties for ease of access
        self.pres_lev = self.props['pres_level']
        self.min_lat = self.props['min_lat']

        # Initialise latitude & theta output arrays with correct shape
        dims = self.data.uwnd.shape

        self.jet_lat = np.zeros([2, dims[0]])
        self.jet_intens = np.zeros([2, dims[0]])

        self.time = self.data.time[:]
        self.tix = None
        self.xix = None

    def find_jet(self, shemis=True):
        """
        Find the subtropical jet using input parameters.

        Parameters
        ----------
        shemis : logical, optional
            If True, find jet position in Southern Hemisphere, if False, find N.H. jet

        """
        # Find axis
        lat_axis = self.data.uwnd.shape.index(self.data.lat.shape[0])

        if self.data.uwnd.shape.count(self.data.lat.shape[0]) > 1:
            # Log a message about which matching dimension used since this
            # could be time or lev or lon if ntimes, nlevs or nlons == nlats
            self.log.info('ASSUMING LAT DIM IS: {} ({})'.format(lat_axis,
                                                                self.data.uwnd.shape))

        self.hemis = [slice(None)] * self.data.uwnd.ndim

        if shemis:
            self.hemis[lat_axis] = self.data.lat < 0
            lat = self.data.lat[self.data.lat < 0]
            hidx = 0
        else:
            self.hemis[lat_axis] = self.data.lat > 0
            lat = self.data.lat[self.data.lat > 0]
            hidx = 1

        # Get uwnd on pressure level
        if self.data.uwnd[self.hemis].shape[1] != 1:
            uwnd_p = self.data.uwnd[self.hemis][:, self.data.lev == self.pres_lev, ...]
        else:
            uwnd_p = self.data.uwnd[self.hemis]

        uwnd_p = np.squeeze(uwnd_p)
        dims = uwnd_p.shape

        self.log.info('COMPUTING JET POSITION FOR %d TIMES HEMIS: %d', dims[0], hidx)
        for tix in range(dims[0]):
            if tix % 50 == 0 and dims[0] > 50:
                self.log.info('COMPUTING JET POSITION FOR %d', tix)
            self.tix = tix
            jet_loc = np.zeros(dims[-1])
            for xix in range(dims[-1]):
                self.xix = xix
                jet_loc[xix] = self.find_single_jet(uwnd_p[tix, :, xix])

            jet_lat = np.ma.masked_where(jet_loc == 0, lat[jet_loc.astype(int)])
            self.jet_lat[hidx, tix] = np.ma.mean(jet_lat)

            jet_intens = np.nanmean(uwnd_p[tix, :, :], axis=-1)
            jet_intens = np.ma.masked_where(jet_loc == 0,
                                            jet_intens[jet_loc.astype(int)])
            self.jet_intens[hidx, tix] = np.ma.mean(jet_intens)

    def find_single_jet(self, uwnd):
        """
        Find the position of the maximum zonal wind of a 1D array of zonal wind.

        Parameters
        ----------
        uwnd : array_like
            1D array of zonal wind of the same shape as input latitude.

        Returns
        -------
        u_max_loc : integer
            Integer position of maximum wind (argmax)

        """
        # Yeah, this is really simple, so what? Maybe someday this function grows
        # up to do more than just the argmax, you don't know!
        return np.argmax(uwnd)


class STJKangPolvani(STJMetric):
    """
    Subtropical jet position metric: Kang and Polvani 2010.

    Parameters
    ----------
    props : :py:meth:`~STJ_PV.run_stj.JetFindRun`
        Class containing properties about the current search for the STJ
    data : :py:meth:`~STJ_PV.input_data.InputData`
        Input data class containing a year (or more) of required data
    """

    def __init__(self, props, data):

        """Initialise Metric using Kang and Polvani Method."""

        name = 'KangPolvani'
        super(STJKangPolvani, self).__init__(name=name, props=props, data=data)

        self.dates = pd.DatetimeIndex(num2date(self.data.time[:], self.data.time_units))

        self.jet_intens_daily = np.zeros([2, self.dates.shape[0]])
        # Seasonal mean is expected
        self.jet_lat_daily = np.zeros([2, self.dates.shape[0]])

        # Seasonal and monthly mean positions
        # self.jet_lat_sm = np.zeros([2, 4])
        # self.jet_lat_mm = np.zeros([2, 12])

        # Output monthly means for comparison
        num_mon = len(np.unique(self.dates.year)) * 12
        self.jet_lat = np.zeros([2, num_mon])
        self.jet_intens = np.zeros([2, num_mon])
        self.wh_1000 = None
        self.wh_200 = None


    def find_jet(self, shemis=True):
        """
        Find the subtropical jet using input parameters.

        Parameters
        ----------
        shemis : logical, optional
            If True, find jet position in Southern Hemisphere, if False, find N.H. jet

        """

        lat_elem, hidx = self.set_hemis(shemis)

        uwnd, vwnd = self._prep_data(lat_elem)
        del_f = self.get_flux_div(uwnd, vwnd, lat_elem)
        self.get_jet_lat(del_f, np.mean(uwnd, axis=-1), self.data.lat[lat_elem], hidx)

    def set_hemis(self, shemis):
        """
        Select hemisphere data.

        This function sets `self.hemis` to be an length N list of slices such that only
        the desired hemisphere is selected with N-D data (e.g. uwind and ipv) along all
        other axes. It also returns the latitude for the selected hemisphere, an index
        to select the hemisphere in output arrays, and the extrema function to find
        min/max of PV derivative in a particular hemisphere.

        Parameters
        ----------
        shemis : boolean
            If true - use southern hemisphere data, if false, use NH data

        Returns
        -------
        lat_elem : array_like
            Latitude element locations for given hemisphere
        hidx : int
            Hemisphere index 0 for SH, 1 for NH

        """

        lat_axis = self.data.uwnd.shape.index(self.data.lat.shape[0])

        if self.data.uwnd.shape.count(self.data.lat.shape[0]) > 1:
            # Log a message about which matching dimension used since this
            # could be time or lev or lon if ntimes, nlevs or nlons == nlats
            self.log.info('ASSUMING LAT DIM IS: {} ({})'.format(lat_axis,
                                                                self.data.uwnd.shape))

        self.hemis = [slice(None)] * self.data.uwnd.ndim

        if shemis:
            self.hemis[lat_axis] = self.data.lat < 0
            lat_elem = np.where(self.data.lat < 0)[0]
            # needed to find the seasonal mean jet let from each zero crossing latitude
            hidx = 0
        else:
            self.hemis[lat_axis] = self.data.lat > 0
            lat_elem = np.where(self.data.lat > 0)[0]
            hidx = 1

        return lat_elem, hidx

    def _prep_data(self, lat_elem):

        # Test if pressure is in Pa or hPa
        if self.data.lev.max() < 1100.0:
            self.data.lev = self.data.lev * 100.

        # Only compute flux div at 200hpa
        self.wh_200 = np.where(self.data.lev == 20000.)[0]
        assert len(self.wh_200) != 0, 'Cant find 200 hpa level'

        # Need surface data for calc shear
        self.wh_1000 = np.where(self.data.lev == 100000.)[0]
        assert len(self.wh_1000) != 0, 'Cant find 1000 hpa level'

        uwnd = xr.DataArray(self.data.uwnd[:, :, lat_elem, :],
                            coords=[self.dates,
                                    self.data.lev,
                                    self.data.lat[lat_elem],
                                    self.data.lon],
                            dims=['time', 'pres', 'lat', 'lon'])

        vwnd = xr.DataArray(self.data.vwnd[:, :, lat_elem, :],
                            coords=[self.dates,
                                    self.data.lev,
                                    self.data.lat[lat_elem],
                                    self.data.lon],
                            dims=['time', 'pres', 'lat', 'lon'])

        return uwnd, vwnd

    def get_flux_div(self, uwnd, vwnd, lat_elem):
        """
        Calculate the meridional eddy momentum flux divergence

        """

        lat = self.data.lat[lat_elem]

        k_e = Kinetic_Eddy_Energies(uwnd.values[:, self.wh_200, :, :],
                                    vwnd.values[:, self.wh_200, :, :],
                                    lat, self.props['pres_level'], self.data.lon)
        k_e.get_components()
        k_e.calc_momentum_flux()

        del_f = xr.DataArray(np.squeeze(k_e.del_f),
                             coords=[self.dates, self.data.lat[lat_elem]],
                             dims=['time', 'lat'])
        return del_f

    def get_jet_lat(self, del_f, uwnd, lat, hidx):
        """
        Find the 200hpa zero crossing of the meridional eddy momentum flux divergence

        """

        signchange = ((np.roll(np.sign(del_f), 1) - np.sign(del_f)) != 0).values
        signchange[:, 0], signchange[:, -1] = False, False

        stj_lat = np.zeros(uwnd.shape[0])
        stj_int = np.zeros(uwnd.shape[0])

        for t in range(uwnd.shape[0]):
            shear = (uwnd[t, self.wh_200, signchange[t, :]].values -
                     uwnd[t, self.wh_1000, signchange[t, :]].values)
            stj_lat[t] = lat[signchange[t, :]][np.argmax(shear)]
            stj_int[t] = uwnd[t, self.wh_200[0], np.where(lat == stj_lat[t])[0]].values

        # Output the monthly mean of daily S for comparing the method
        jet_data = xr.DataArray(stj_lat, coords=[self.dates], dims=['time'])
        jet_data_mm = jet_data.resample(freq='MS', dim='time')
        self.jet_lat[hidx, :] = jet_data_mm.values

        jet_data = xr.DataArray(stj_int, coords=[self.dates], dims=['time'])
        jet_data_mm = jet_data.resample(freq='MS', dim='time')
        self.jet_intens[hidx, :] = jet_data_mm.values

        dtimes = [dtime.to_pydatetime() for dtime in jet_data_mm.time.to_index()]
        self.time = date2num(dtimes, self.data.time_units, self.data.calendar)

    def get_jet_loc(self, data, expected_lat, lat):
        """Get jet location based on sign changes of Del(f)."""
        signchange = ((np.roll(np.sign(data), 1) - np.sign(data)) != 0).values
        idx = (np.abs(lat[signchange] - expected_lat)).argmin()

        return lat[signchange][idx]

    def loop_jet_lat(self, data, expected_lat, lat):
        """Get jet location at multiple times."""
        return np.array([self.get_jet_loc(data[tidx, :], expected_lat[tidx], lat)
                         for tidx in range(data.shape[0])])


def get_season(month):
    """Map month index to index of season [DJF -> 0, MAM -> 1, JJA -> 2, SON -> 3]."""
    seasons = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])
    return seasons[month]
