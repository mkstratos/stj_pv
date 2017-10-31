# -*- coding: utf-8 -*-
"""STJ Metric: Calculate the position of the subtropical jet in both hemispheres."""
import numpy as np
import numpy.polynomial as poly
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
        dio.write_to_netcdf(out_vars, self.props['output_file'] + '.nc')

    def append(self, other):
        """Append another metric's latitude and theta positon to this one."""
        self.jet_lat = np.append(self.jet_lat, other.jet_lat, axis=1)
        self.jet_intens = np.append(self.jet_intens, other.jet_intens, axis=1)
        if self.jet_theta is not None:
            self.jet_theta = np.append(self.jet_theta, other.jet_theta, axis=1)
        self.time = np.append(self.time, other.time, axis=0)


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
            :py:meth:`scipy.signal.argrelmax` if SH, or :py:meth:`scipy.signal.argrelmin`
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

    def isolate_pv(self, pv_lev):
        """
        Get the potential temperature, zonal wind and zonal wind shear for a PV level.

        Parameters
        ----------
        pv_lev : float
            PV value (for a particular hemisphere, >0 for NH, <0 for SH) on which to
            interpolate potential temperature and wind

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
        theta_xpv = utils.vinterp(self.data.th_lev, self.data.ipv[self.hemis], pv_lev)
        uwnd_xpv = utils.vinterp(self.data.uwnd[self.hemis], self.data.ipv[self.hemis],
                                 pv_lev)
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
        theta_xpv, uwnd_xpv, ushear = self.isolate_pv(pv_lev)
        dims = theta_xpv.shape

        self.log.info('COMPUTING JET POSITION FOR %d TIMES HEMIS: %d', dims[0], hidx)
        for tix in range(dims[0]):
            if tix % 50 == 0 and dims[0] > 50:
                self.log.info('COMPUTING JET POSITION FOR %d', tix)
            self.tix = tix
            jet_loc = np.zeros(dims[-1])
            for xix in range(dims[-1]):
                self.xix = xix
                jet_loc[xix] = self.find_single_jet(theta_xpv[tix, :, xix],
                                                    lat, ushear[tix, :, xix], extrema)
                if not self.props['zonal_opt'].lower() == 'mean':
                    self.jet_lat[hidx, tix, xix] = lat[jet_loc[xix]]
                    self.jet_theta[hidx, tix, xix] = theta_xpv[tix, jet_loc[xix], xix]
                    self.jet_intens[hidx, tix, xix] = uwnd_xpv[tix, jet_loc[xix], xix]

            if self.props['zonal_opt'].lower() == 'mean':

                jet_lat = np.ma.masked_where(jet_loc == 0, lat[jet_loc.astype(int)])
                self.jet_lat[hidx, tix] = np.ma.median(jet_lat)

                # First take the zonal median of Theta on dyn. tropopause
                jet_theta = np.nanmedian(theta_xpv[tix, :, :], axis=-1)
                # Mask wherever jet_loc is undefined, jet_loc is a func. of longitude here
                jet_theta = np.ma.masked_where(jet_loc == 0,
                                               jet_theta[jet_loc.astype(int)])
                # Then save the zonal median of this to the correct position in output
                self.jet_theta[hidx, tix] = np.ma.median(jet_theta)

                # Now do the same for jet intensity
                jet_intens = np.nanmedian(uwnd_xpv[tix, :, :], axis=-1)
                jet_intens = np.ma.masked_where(jet_loc == 0,
                                                jet_intens[jet_loc.astype(int)])
                self.jet_intens[hidx, tix] = np.ma.median(jet_intens)

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
            1D array of latitude same shape as theta_xpv and ttrop
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
        y_e = None

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
        uwnd_p = self.data.uwnd[self.hemis][:, self.data.lev == self.pres_lev, ...]
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
            self.jet_lat[hidx, tix] = np.ma.median(jet_lat)

            jet_intens = np.nanmedian(uwnd_p[tix, :, :], axis=-1)
            jet_intens = np.ma.masked_where(jet_loc == 0,
                                            jet_intens[jet_loc.astype(int)])
            self.jet_intens[hidx, tix] = np.ma.median(jet_intens)

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
