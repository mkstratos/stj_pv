# -*- coding: utf-8 -*-
"""STJ Metric: Calculate the position of the subtropical jet in both hemispheres."""
import numpy as np
import numpy.polynomial as poly
from scipy import interpolate
from scipy import signal as sig

import utils
import data_out as dio


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
        self.time = None

    def save_jet(self):
        """Save jet position to file."""
        # Create output variables
        props_lat = {'name': 'jet_latitude', 'descr': 'Latitude of subtropical jet',
                     'units': 'degrees_north', 'short_name': 'lat_sh', 'timevar': 'time',
                     'calendar': self.data.calendar, 'time_units': self.data.time_units}
        coords = {'time': self.time}

        if self.props['zonal_opt'].lower() != 'mean':
            props_lat['lonvar'] = 'lon'
            props_lat['lon_units'] = 'degrees_east'
            coords['lon'] = self.data.lon

        props_th = dict(props_lat)
        props_th['name'] = 'jet_theta'
        props_th['descr'] = 'Theta level of subtropical jet'
        props_th['units'] = 'K'
        props_th['short_name'] = 'theta_sh'

        self.log.info("CREATE VARIABLES")
        lat_sh_out = dio.NCOutVar(self.jet_lat[0, ...], coords=coords, props=props_lat)
        theta_sh_out = dio.NCOutVar(self.jet_theta[0, ...], coords=coords, props=props_th)
        props_lat_nh = dict(props_lat)
        props_th_nh = dict(props_th)
        props_lat_nh['short_name'] = 'lat_nh'
        props_th_nh['short_name'] = 'theta_nh'
        lat_nh_out = dio.NCOutVar(self.jet_lat[1, ...], coords=coords, props=props_lat_nh)
        theta_nh_out = dio.NCOutVar(self.jet_theta[1, ...], coords=coords,
                                    props=props_th_nh)

        self.log.info("WRITE TO {output_file}".format(**self.props))
        # Write jet/theta positions to file
        dio.write_to_netcdf([lat_sh_out, theta_sh_out, lat_nh_out, theta_nh_out],
                            self.props['output_file'] + '.nc')

    def append(self, other):
        """Append another metric's latitude and theta positon to this one."""
        self.jet_lat = np.append(self.jet_lat, other.jet_lat, axis=1)
        self.jet_theta = np.append(self.jet_theta, other.jet_theta, axis=1)
        self.time = np.append(self.time, other.time, axis=0)


class STJPV(STJMetric):
    """Subtropical jet position metric using dynamic tropopause on isentropic levels."""

    def __init__(self, jet_run, data):
        """
        Initialise Metric using PV Gradient Method.

        Parameters
        ----------
        jet_run : JetFindRun
            Class containing properties about the current search for the STJ
        data : InputData
            Input data class containing a year (or more) of required data

        """
        name = 'PVGrad'
        super().__init__(name=name, props=jet_run, data=data)

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

        # Initialise latitude/theta output arrays with correct shape
        dims = self.data.ipv.shape
        if self.props['zonal_opt'].lower() == 'mean':
            self.jet_lat = np.zeros([2, dims[0]])
            self.jet_theta = np.zeros([2, dims[0]])
        else:
            self.jet_lat = np.zeros([2, dims[0], dims[-1]])
            self.jet_theta = np.zeros([2, dims[0], dims[-1]])

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

        return poly_der

    def find_jet(self, shemis=True):
        """
        Find the subtropical jet using input parameters.

        Parameters
        ----------
        shemis : logical, optional
            If True, find jet position in Southern Hemisphere, if False, find N.H. jet

        """
        if shemis and self.pv_lev < 0 or not shemis and self.pv_lev > 0:
            pv_lev = self.pv_lev
        else:
            pv_lev = -self.pv_lev

        # Find axis
        lat_axis = self.data.ipv.shape.index(self.data.lat.shape[0])
        lat_axis_3d = self.data.trop_theta.shape.index(self.data.lat.shape[0])

        if self.data.ipv.shape.count(self.data.lat.shape[0]) > 1:
            # Log a message about which matching dimension used since this
            # could be time or lev or lon if ntimes, nlevs or nlons == nlats
            self.log.info('ASSUMING LAT DIM IS: {} ({})'.format(lat_axis,
                                                                self.data.ipv.shape))

        hem_slice = [slice(None)] * self.data.ipv.ndim
        hem_slice_3d = [slice(None)] * self.data.trop_theta.ndim

        if shemis:
            hem_slice[lat_axis] = self.data.lat < 0
            hem_slice_3d[lat_axis_3d] = self.data.lat < 0
            lat = self.data.lat[self.data.lat < 0]
            hidx = 0
            # Link `extrema` function to argrelmax for SH
            extrema = sig.argrelmax
        else:
            hem_slice[lat_axis] = self.data.lat > 0
            hem_slice_3d[lat_axis_3d] = self.data.lat > 0
            lat = self.data.lat[self.data.lat > 0]
            hidx = 1
            # Link `extrema` function to argrelmin for NH
            extrema = sig.argrelmin

        # Get theta on PV==pv_level
        theta_xpv = utils.vinterp(self.data.th_lev, self.data.ipv[hem_slice],
                                  np.array([pv_lev]))
        dims = theta_xpv.shape

        uwnd = self.data.uwnd[hem_slice]
        ttrop = self.data.trop_theta[hem_slice_3d]

        self.log.info('COMPUTING JET POSITION FOR {} TIMES'.format(dims[0]))
        for tix in range(dims[0]):
            self.tix = tix
            jet_loc = np.zeros(dims[-1])
            for xix in range(dims[-1]):
                self.xix = xix
                jet_loc[xix] = self._find_single_jet(theta_xpv[tix, :, xix],
                                                     ttrop[tix, :, xix],
                                                     lat, uwnd[tix, ..., xix], extrema)
                if not self.props['zonal_opt'].lower() == 'mean':
                    self.jet_lat[hidx, tix, xix] = lat[jet_loc[xix]]
                    self.jet_theta[hidx, tix, xix] = theta_xpv[tix, jet_loc[xix], xix]

            if self.props['zonal_opt'].lower() == 'mean':
                jet_lat = np.ma.masked_where(jet_loc == 0, lat[jet_loc.astype(int)])
                jet_theta = np.nanmean(theta_xpv[tix, :, :], axis=-1)
                jet_theta = np.ma.masked_where(jet_loc == 0,
                                               jet_theta[jet_loc.astype(int)])

                self.jet_lat[hidx, tix] = np.ma.mean(jet_lat)
                self.jet_theta[hidx, tix] = np.ma.mean(jet_theta)

    def _find_single_jet(self, theta_xpv, ttrop, lat, uwnd, extrema):
        """
        Find jet location for a 1D array of theta on latitude.

        Parameters
        ----------
        theta_xpv : array_like
            Theta on PV level as a function of latitude
        ttrop : array_like
            Thermal tropopause theta as a function of latitude
        lat : array_like
            1D array of latitude same shape as theta_xpv and ttrop
        uwnd : array_like
            2D array of wind (Height x Latitude) on theta levels

        Returns
        -------
        jet_loc : int
            Index of jet location on latitude axis

        """
        # Get thermal tropopause intersection with dynamical tropopause within 45deg
        # of the equator
        y_s = np.abs(np.ma.masked_invalid(ttrop[np.abs(lat) < 45] -
                                          theta_xpv[np.abs(lat) < 45])).argmin()
        y_e = None

        # If latitude is in decreasing order, switch start/end
        # This makes sure we're selecting the latitude nearest the equator
        if lat[0] < lat[-1]:
            y_s, y_e = y_e, y_s

        # Find derivative of dynamical tropopause
        dtheta = self._poly_deriv(lat, theta_xpv, y_s=y_s, y_e=y_e)

        jet_loc_all = extrema(dtheta)[0].astype(int)
        if y_s is not None:
            # If beginning of array is cut off rather than end, add cut-off to adjust
            jet_loc_all += y_s

        return self.select_jet(jet_loc_all, lat, uwnd)

    def select_jet(self, locs, lat, uwnd):
        """Select correct jet latitude."""
        if len(locs) == 0:
            self.log.info("NO JET LOC t:{} lon:{}".format(self.tix, self.xix))
            jet_loc = 0

        elif len(locs) == 1:
            # This essentially converts a list of length 1 to a single int
            jet_loc = locs[0]

        elif len(locs) > 1:
            # TODO: Decide on which (if multiple local mins) to be the location
            # Should be based on wind shear at that location, for now, lowest latitude
            jet_loc = locs[np.abs(lat[locs]).argmin()]

        return jet_loc
