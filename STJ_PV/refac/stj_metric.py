"""STJ Metric: Calculate the position of the subtropical jet in both hemispheres."""
import pdb
import numpy as np
# from numpy.polynomial import chebyshev as cby
import numpy.polynomial as poly
from scipy import interpolate
from scipy import signal as sig

import input_data
import calc_ipv as cpv


class MetricData(object):
    """Container class for STJ PV Metric data."""

    def __init__(self):
        """Initalise container class for STJ PV Metric data."""
        self.actual_ipv_values = {}
        self.theta_cby_val = {}


class MetricResult(object):
    """Container class for STJ PV metric results data."""

    def __init__(self, shape):
        """Initialize MetricResults class with pre-allocated arrays for metric results."""

        ntimes = shape['time']
        ntheta = shape['theta']
        nlon = shape['lon']

        self.hemis = ['nh', 'sh']

        # Hemispherically separate to produce jet metrics for each hemisphere
        # [time, hemisphere, theta in restriced domain]
        self.theta_xpv = {hem: np.full([ntimes, ntheta], np.nan) for hem in self.hemis}
        self.dtdphi = {hem: np.full([ntimes, ntheta], np.nan) for hem in self.hemis}

        # STJ metric [time_loop, lon_loop, hemi_count, cby or fd]
        self.jet_best_guess = {hem: np.full([ntimes, nlon, 2], np.nan)
                               for hem in self.hemis}

        # point where two tropopause definitions cross paths
        self.crossing_lat = {hem: np.zeros(ntimes) for hem in self.hemis}

        # [time_loop, lon_loop, hemi_count, cby or fd]
        self.jet_intensity = {hem: np.full([ntimes, nlon, 2], np.nan)
                              for hem in self.hemis}

        # [time_loop, lon_loop, hemi_count, cby or fd]
        self.jet_th_lev = {hem: np.full([ntimes, nlon, 2], np.nan) for hem in self.hemis}


class STJIPVMetric(object):
    """Class to compute and store information about subtropical jet."""

    def __init__(self, props):
        """
        Initialise STJ Metric calculations.

        Parameters
        ----------
        props : STJProperties
            Properties of the STJ Metric run
        """
        self.props = props
        self.in_data = input_data.InputData(self.props)
        self.pv_lev = self.props.run_opts['pv_value']
        self.data = None
        self.elem_loc = None
        self.ipv_zonal = {}
        self.ipv_dom = {}
        self.theta_dom = {}
        self.lat_dom = {}
        self.phi_xpv = None
        self.theta_xpv = None

    def calc_metric(self):
        """Calculate the position of the subtropical jet."""
        # Set a boolean flag for zonal mean / longitude slices
        zonal_mean_flag = (self.props.run_opts['slicing'] == 'zonal_mean')
        self._setup_data()
        # Create MetricResult object using correct shapes, from input data (times, lon)
        # and interpolation arrays (theta)
        self.data = MetricResult({'time': self.in_data.ipv.shape[0],
                                  'theta': self.theta_dom['nh'].shape[0],
                                  'lon': self.elem_loc.shape[0]})
        for hemi in self.data.hemis:
            self._interp_pv(hemi, zonal_mean_flag)
        self._pv_deriv(hemi)

    def _setup_data(self):
        """Setup data depending on `self.props` for use within calc_metric."""
        # Prep step 1: Define lat interpolate IPV to
        lat_increment = self.props.run_opts['dlat']
        self.lat_dom = {'sh': np.arange(-90, 0 + lat_increment, lat_increment),
                        'nh': np.arange(0, 90 + lat_increment, lat_increment)}

        # Prep step 2: Define theta interpolate IPV to
        # using a smaller domain to avoid areas where 2.0 PV is not defined
        theta_increment = self.props.run_opts['dtheta']
        self.theta_dom['nh'] = np.arange(310, 401, theta_increment)
        self.theta_dom['sh'] = np.arange(310, 401, theta_increment)

        # Prep step 3: Define slices of longitude to be used
        if self.in_data.lon.min() == 0.0 and self.props.run_opts['nslice'] > 0:
            lon_slice = np.linspace(0, 360.0, self.props.run_opts['nslice'])

        elif self.in_data.lon.min() == -180.0 and self.props.run_opts['nslice'] > 0:
            lon_slice = np.linspace(-180.0, 180.0, self.props.run_opts['nslice'])

        elif self.props.run_opts['nslice'] == 0:
            lon_slice = np.array([0.0])

        else:
            # Data isn't global...use local min/max
            lon_slice = np.linspace(-self.in_data.lon.min(), self.in_data.lon.max(),
                                    self.props.run_opts['nslice'])

        # Generate a meshgrid so we can do a subtraction between data's lon and lon slices
        lon_2d, lon_slice_2d = np.meshgrid(self.in_data.lon, lon_slice)

        # Need to find index of data's longitude nearest to the slices
        self.elem_loc = np.abs(lon_2d - lon_slice_2d).argmin(axis=0)

    def _interp_pv(self, hemi, zonal_mean=True):
        """Fit polynomial to specific PV contour on theta surfaces as fcn of latitude."""

        # Step 1: Take the zonal mean, if desired
        if zonal_mean:
            # Gives [time, theta, lat], have to use nanmean, in case some data == nan
            self.ipv_zonal[hemi] = np.nanmean(self.in_data.ipv, axis=-1)

            # Step 2: Interpolate zonal mean onto new lat and theta
            self.ipv_dom[hemi] = interp_nd(self.in_data.lat, self.in_data.th_levels,
                                           self.ipv_zonal, self.lat_dom[hemi],
                                           self.theta_dom[hemi])
        else:
            # Step 2: Interpolate full field onto new lat and theta
            self.ipv_dom[hemi] = interp_nd(self.in_data.lat, self.in_data.th_levels,
                                           self.in_data.ipv, self.lat_dom[hemi],
                                           self.theta_dom[hemi])

        # Step 3: Find the element location closest to +-2.0 PV line
        # - code for monotonic increase theta and phi.
        # - actual_ipv_values is used to locic test the 2pv line
        # - phi_idx_xpv is the element locations where the near-2.0 pv element occurs.
        self.theta_xpv = cpv.vinterp(self.in_data.th_levels, self.ipv_dom[hemi],
                                     np.array([self.pv_lev]))

    def _pv_deriv(self, degree=12):
        """
        Calculate the Chebyshev polynomial fit and derivatives of a PV contour.

        First compute the polynomial fit, then first two derivatives of the contour using
        a Chebyshev polynomial of degree 10.

        Parameters
        ----------
        degree : integer
            Degree of Chebyshev polynomial to use for fit (more is more accurate but more
            expensive)
        """
        for hemi in self.data.hemis:
            # Find the chebyshev polynomial fit
            theta_cby = cby.chebfit(self.data.phi_xpv[hemi],
                                    self.data.theta_xpv[hemi], degree)

            # then differentiate dtheta_2PV/dy and d^2(theta_2PV)/dy^2
            dtdphi_cby = cby.chebder(theta_cby)
            d2tdphi2_cby = cby.chebder(dtdphi_cby)

            # Values of the fit
            self.data.theta_cby_val[hemi] = cby.chebval(self.data.phi_xpv[hemi],
                                                        theta_cby)

            # Values of the derivative d theta / d phi
            self.data.dtdphi[hemi] = cby.chebval(self.data.phi_xpv[hemi],
                                                 dtdphi_cby)

            # Values of the second derivative d^2 (theta) / d phi^2
            self.data.d2tdphi2_val[hemi] = cby.chebval(self.data.phi_xpv[hemi],
                                                       d2tdphi2_cby)

        # if time_loop == 73:
        #  testing = True
        # else:
        #  testing = False
        # testing = False
        # test the poly fit
        # if testing:
        #    Poly_testing(self.phi_2PV, self.theta_2PV, self.theta_cby_val,
        #                 self.dtdphi, self.d2tdphi2_val)
        #    pdb.set_trace()

    def thermal_trop_theta(self):
        """Calculate the location of the thermal tropopause in theta coordinates."""

    def unique_theta_pv(self, hemi):
        """Find and eliminate duplicate lat/theta coordinate for PV contour."""

        # Remove repeated elements.
        phi_xpv, phi_idx = np.unique(self.phi_xpv, return_index=True)

        # sort the elements to ensure the array order is the same then apply to all arrays
        phi_idx = (np.sort(phi_idx)).tolist()

        # test that each 2PV point on the line is increasing in phi for NH (visa versa).
        i = 0
        while i <= (len(self.phi_xpv[phi_idx]) - 2):
            dphi = self.phi_xpv[phi_idx][i + 1] - self.phi_xpv[phi_idx][i]
            if hemi == 'NH' and dphi > 0:
                # remove the element i
                phi_idx.remove(phi_idx[i + 1])
                i = i - 1
            if hemi == 'SH' and dphi < 0:
                phi_idx.remove(phi_idx[i])
                i = i - 1
            i = i + 1

        # data
        self.phi_xpv = self.phi_xpv[phi_idx]
        self.theta_xpv = self.theta_xpv[phi_idx]

        # test if there are two d_theta values that are the same next to each
        # other and if so remove one.
        theta_xpv_test1, idx_test1, idx_test2 = np.unique(
            self.theta_xpv, return_inverse=True, return_index=True)

        if len(self.theta_xpv) != len(theta_xpv_test1):
            # catch if the dimensions are different
            print('Dimensions of theta_xpv and theta_xpv_test1 are different')
            pdb.set_trace()

    def diff_2pv_fd(self):
        """Differentiate PV contour using finite differences."""

    def find_peaks(self):
        """Find turning points in derivative of PV contour."""

    def sort_peaks(self):
        """Sort peaks in derivative by latitude."""

    def max_shear(self):
        """Find shear of all turning points, return point with maximum wind shear."""

    def save_output(self):
        """Save output generated by metric method to a file."""


def get_pv_lev(pv_data, pv_lev):
    """
    Get the `pv_lev` contour latitude and value (should be ~= pv_lev).

    Parameters
    ----------
    pv_data : array_like
        Input IPV data on (theta, latitude), (time, theta, latitude) or
        (time, theta, latitude, longitude) grid
    pv_lev : float
        Desired PV contour in same units as `pv_data`

    Returns
    -------
    lat_idx : array_like
        Latitude of `pv_lev` contour at each theta level (same shape as `pv_data.shape[0]`
    pv_idx_vals : array_like
        Actual PV Values nearest to pv_lev
    """
    if pv_data.ndim == 3 or pv_data.ndim == 2:
        lat_axis = -1
    elif pv_data.ndim == 4:
        lat_axis = -2
    else:
        raise NotImplementedError('No method for {} PV dimensions'.format(pv_data.ndim))

    pv_shape = list(pv_data.shape)
    pv_shape.pop(lat_axis)

    lat_idx = np.abs(pv_data - pv_lev).argmin(axis=lat_axis)
    all_idx = np.indices(pv_shape).tolist()
    all_idx.insert(pv_data.ndim + lat_axis, lat_idx)

    return lat_idx, pv_data[all_idx]


def interp_nd(lat, theta, data, lat_hr, theta_hr):
    """
    Perform interpolation on 2-dimensions on up to 4-dimensional numpy array.

    Parameters
    ----------
    lat : array_like
        One dimensional latitude coordinate array, matches a dimension of `data`
    theta : array_like
        One dimensional theta (vertical) coordinate array, matches a dimension of `data`
    data : array_like
        Data to be interpolated to high-resolution grid
    lat_hr : array_like
        1-D latitude coordinate array that `data` is interpolated to
    theta_hr : array_like
        1-D vertical coordinate array that `data` is interpolated to

    Returns
    -------
    data_interp : array_like
        Interpolated data where lat/theta dimensions are interpolated to `lat_hr` and
        `theta_hr`
    """
    lat_dim = np.where(np.array(data.shape) == lat.shape[0])[0]
    theta_dim = np.where(np.array(data.shape) == theta.shape[0])[0]

    if data.ndim == 2:
        data_f = interpolate.interp2d(lat, theta, data, kind='cubic')
        data_interp = data_f(lat_hr, theta_hr)

    elif data.ndim == 3:
        out_shape = list(data.shape)
        out_shape[lat_dim] = lat_hr.shape[0]
        out_shape[theta_dim] = theta_hr.shape[0]

        data_interp = np.zeros(out_shape)
        cmn_axis = np.where(out_shape == np.array(data.shape))[0]

        for idx0 in range(data.shape[cmn_axis]):
            data_f = interpolate.interp2d(lat, theta, data.take(idx0, axis=cmn_axis),
                                          kind='cubic')
            slc = [slice(None)] * data_interp.ndim
            slc[cmn_axis] = idx0
            data_interp[slc] = data_f(lat_hr, theta_hr)

    elif data.ndim == 4:

        out_shape = list(data.shape)
        out_shape[lat_dim] = lat_hr.shape[0]
        out_shape[theta_dim] = theta_hr.shape[0]
        data_interp = np.zeros(out_shape)

        cmn_axis = np.where(out_shape == np.array(data.shape))[0][:]
        for idx0 in range(data.shape[cmn_axis[0]]):
            for idx1 in range(data.shape[cmn_axis[1]]):
                data_slice = data.take(idx1, axis=cmn_axis[1]).take(idx0,
                                                                    axis=cmn_axis[0])
                data_f = interpolate.interp2d(lat, theta, data_slice, kind='cubic')
                # slc says which axis to place interpolated array on, it's what changes
                # with the loops
                slc = [slice(None)] * data_interp.ndim
                slc[cmn_axis[0]] = idx0
                slc[cmn_axis[1]] = idx1
                data_interp[slc] = data_f(lat_hr, theta_hr)
    return data_interp


class STJPV(object):
    """
    Metric for Subtropical jet position using dynamic tropopause on isentropic levels.
    """
    name = 'PVGrad'

    def __init__(self, props, data):
        self.props = props
        self.data = data
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

        # Initialise latitude/theta output arrays with correct shape
        dims = self.data.ipv.shape
        if self.props['zonal_opt'] == 'mean':
            self.jet_lat = np.zeros([2, dims[0]])
            self.jet_theta = np.zeros([2, dims[0]])
        else:
            self.jet_lat = np.zeros([2, dims[0], dims[-1]])
            self.jet_theta = np.zeros([2, dims[0], dims[-1]])

    def _poly_deriv(self, data, y_s=None, y_e=None, deriv=1):
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
        poly_fit = self.pfit(self.data.lat[y_s:y_e], data[y_s:y_e], self.fit_deg)
        poly_der = self.peval(self.data.lat[y_s:y_e], self.pder(poly_fit, deriv))

        return poly_der

    def find_jet(self, shemis=True):
        """
        Using input parameters find the subtropical jet.

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

        if self.data.ipv.shape.count(self.lat.shape[0]) > 1:
            # Print a message about which matching dimension used since this
            # could be time or lev if ntimes == nlats or nlevs == nlats
            print('ASSUMING LAT DIM IS: {} ({})'.format(lat_axis, self.data.ipv.shape))

        hem_slice = [slice(None)] * self.data.ipv.ndim
        hem_slice_3d = [slice(None)] * self.data.trop_theta.ndim

        if shemis:
            hem_slice[lat_axis] = self.data.lat < 0
            hem_slice_3d[lat_axis_3d] = self.data.lat < 0
            hidx = 0
            extrema = sig.argrelmax
        else:
            hem_slice[lat_axis] = self.data.lat > 0
            hem_slice_3d[lat_axis_3d] = self.data.lat > 0
            hidx = 1
            extrema = sig.argrelmin

        # Get theta on PV==pv_level
        theta_xpv = cpv.vinterp(self.data.th_lev, self.data.ipv[hem_slice],
                                np.array([pv_lev]))
        dims = theta_xpv.shape

        uwnd = self.data.uwnd[hem_slice]
        ttrop = self.data.trop_theta[hem_slice_3d]

        if self.props['zonal_opt'] == 'mean':
            # Zonal mean stuff
            theta_xpv = np.nanmean(theta_xpv, axis=-1)
            uwnd = np.nanmean(uwnd, axis=-1)
            ttrop = np.nanmean(ttrop, axis=-1)

        for tix in range(dims[0]):

            # Get thermal tropopause intersection with dynamical tropopause
            y_s = np.abs(self.data.trop_theta[tix, :] - theta_xpv[tix, :]).argmin()
            y_e = None

            # If latitude is in decreasing order, switch start/end
            if self.data.lat[0] > self.data.lat[-1]:
                y_s, y_e = y_e, y_s

            # Find derivative of dynamical tropopause
            dtheta = self._poly_deriv(theta_xpv[tix, :], y_s=y_s, y_e=y_e)

            jet_loc_all = extrema(dtheta)[0].astype(int)
            if y_s is not None:
                # If beginning of array is cut off rather than end, add cut-off to adjust
                jet_loc_all += y_s

            jet_loc = self.select_jet(jet_loc_all, tix, uwnd[tix, ...])

            self.jet_lat[hidx, tix] = self.data.lat[tix, jet_loc]
            self.jet_theta[hidx, tix] = theta_xpv[tix, jet_loc]

    def select_jet(self, locs, tix, uwnd):
        """Select correct jet latitude."""
        if len(locs) == 0:
            self.props.log.info("NO JET LOC {}".format(tix))
            jet_loc = 0

        elif len(locs) == 1:
            jet_loc = locs[0]

        elif len(jet_loc) > 1:
            # TODO: Decide on which (if multiple local mins) to be the location
            # Should be based on wind shear at that location, for now, lowest latitude
            jet_loc = locs[np.abs(self.data.lat[locs]).argmin()]

        return jet_loc

    def save_jet(self):
        """
        Save jet position to file.
        """
        # Create output variables
        # Write jet/theta positions to file
