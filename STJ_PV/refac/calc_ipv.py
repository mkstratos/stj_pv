# -*- coding: utf-8 -*-
"""
Methods for calculating isentropic potential vorticity from data on pressure levels.
"""

import numpy as np


# specify the range and increment over which to calculate IPV
TH_LEV = np.arange(300, 501, 5)
RAD = np.pi / 180.0  # radians per degree
OM = 7.292e-5  # Angular rotation rate of earth    [rad]
GRV = 9.81      # Acceleration due to GRVity       [m/s^2]
EARTH_R = 6.371e6

__author__ = "Michael Kelleher"


class NDSlicer(object):
    """N-Dimensional slice class for numpy arrays."""
    def __init__(self, axis, ndim, start=None, end=None, skip=None):
        """
        Create an n-dimensional slice list.

        Parameters
        ----------
        axis : integer
            Axis on which to apply the slice
        ndim : integer
            Total number of dimensions of array to be sliced
        start, end, skip : integer, optional
            Index of beginning, end and skip width of the slice [start:end:skip]
            default for each is None.
        """
        self.axis = axis
        self.ndim = ndim
        self.start = start
        self.end = end
        self.skip = skip
        self.slicer = None
        self.slice(start, end, skip)

    def slice(self, start=None, end=None, skip=None):
        """
        Create an n-dimensional slice list.

        Parameters
        ----------
        axis : integer
            Axis on which to apply the slice
        ndim : integer
            Total number of dimensions of array to be sliced
        start, end, skip : integer, optional
            Index of beginning, end and skip width of the slice [start:end:skip]
            default for each is None.

        Returns
        -------
        slicer : list
            list of slices such that all data at other axes are kept, one axis is sliced

        Examples
        --------
        x = np.random.randn(5, 3)

        # Create slicer equivalent to [1:-1, :]
        slc = NDSlicer(0, x.ndim)
        print(x)
        [[ 0.68470539  0.87880216 -0.45086367]
         [ 1.06804045  0.63094676 -0.76633033]
         [-1.69841915  0.35207064 -0.4582049 ]
         [-0.56431067  0.62833728 -0.04101542]
         [-0.02760744  2.02814338  0.13195714]]
        print(x[slc.slice(1, -1)])
        [[ 1.06804045  0.63094676 -0.76633033]
         [-1.69841915  0.35207064 -0.4582049 ]
         [-0.56431067  0.62833728 -0.04101542]]
        """
        self.start = start
        self.end = end
        self.skip = skip
        self.slicer = [slice(None)] * self.ndim
        self.slicer[self.axis] = slice(self.start, self.end, self.skip)
        return self.slicer


def diff_cfd(data, axis=-1, cyclic=False):
    """
    Calculate centered finite difference on a field along an axis with even spacing.

    Parameters
    ----------
    data : array_like
        ND array of data of which to calculate the differences
    axis : integer
        Axis of `data` on which differences are calculated
    cyclic : bool
        Flag to indicate whether `data` is cyclic on `axis`

    Returns
    -------
    diff : array_like
        ND array of central finite differences of `data` along `axis`
    """
    # Calculate centred differences along longitude direction
    # Eqivalent to: diff = data[..., 2:] - data[..., :-2] for axis == -1
    slc = NDSlicer(axis, data.ndim)
    diff = data[slc.slice(2, None)] - data[slc.slice(None, -2)]

    if cyclic:
        # Cyclic boundary in "East"
        # Equiv to diff[..., 0] = data[..., 1:2] - data[..., -1:]
        d_1 = (data[slc.slice(1, 2)] - data[slc.slice(-1, None)])
        diff = np.append(d_1, diff, axis=axis)

        # Cyclic boundary in "West"
        # Equiv to diff[..., -1] = data[..., 0:1] - data[..., -2:-1]
        diff = np.append(diff, (data[slc.slice(0, 1)] - data[slc.slice(-2, -1)]),
                         axis=axis)
    else:
        # Otherwise edges are forward/backward differences
        # Boundary in "South", (data[..., 1:2] - data[..., 0:1])
        diff = np.append((data[slc.slice(1, 2)] - data[slc.slice(0, 1)]), diff, axis=axis)

        # Boundary in "North" (data[..., -1:] - data[..., -2:-1])
        diff = np.append(diff, (data[slc.slice(-1, None)] - data[slc.slice(-2, -1)]),
                         axis=axis)

    return diff


def diffz(data, vcoord, axis=None):
    """
    Calculate vertical derivative for data on uneven vertical levels.

    Parameters
    ----------
    data : array_like
        N-D array of input data to be differentiated, where
        data.shape[axis] == vcoord.shape[0]
    vcoord : array_like
        Vertical coordinate, 1D
    axis : integer, optional
        Axis where data.shape[axis] == vcoord.shape[0]

    Returns
    -------
    dxdz : array_like
        N-D array of d(data)/d(vcoord), same shape as input `data`
    """
    if axis is None:
        # Find matching axis between data and vcoord
        axis = data.shape.index(vcoord.shape[0])

    # Create array to hold vertical derivative
    dxdz = np.ones(data.shape)

    # Create n-dimensional slicer along matching axis
    slc = NDSlicer(axis, data.ndim)

    # Create an n-dimensional broadcast along matching axis, same as [None, :, None, None]
    # for axis=1, ndim=4
    bcast = [np.newaxis] * data.ndim
    bcast[axis] = slice(None)

    d_z = (vcoord[1:] - vcoord[:-1])
    d_z2 = d_z[:-1][bcast]
    d_z1 = d_z[1:][bcast]

    dxdz[slc.slice(1, -1)] = ((d_z2 * data[slc.slice(2, None)] +
                               (d_z1 - d_z2) * data[slc.slice(1, -1)] -
                               d_z1 * data[slc.slice(None, -2)]) /
                              (2.0 * d_z1 * d_z2))

    # Do forward difference at 0th level [:, 1, :, :] - [:, 0, :, :]
    dz1 = vcoord[1] - vcoord[0]
    dxdz[slc.slice(0, 1)] = (data[slc.slice(0, 1)] - data[slc.slice(1, 2)]) / dz1

    # Do backward difference at Nth level
    dz1 = vcoord[-1] - vcoord[-2]
    dxdz[slc.slice(-1, None)] = (data[slc.slice(-1, None)] -
                                 data[slc.slice(-2, -1)]) / dz1

    return dxdz


def vinterp(data, vcoord, vlevels):
    """
    Perform linear vertical interpolation.

    Parameters
    ----------
    data : array_like (>= 2D)
        Array of data to be interpolated
    vcoord : array_like (>= 2D, where data.shape[1] == vcoord.shape[1])
        Vertical coordinate to interpolate to
    vlevels : array_like (1D)
        Levels, in same units as vcoord, to interpolate to

    Returns
    -------
    out_data : array_like, (data.shape[0], vlevels.shape[0], *data.shape[2:])
        Data on vlevels
    """
    vcoord_shape = list(vcoord.shape)
    vcoord_shape.pop(1)
    if np.sum(vcoord[:, 0, ...] > vcoord[:, -1, ...]) / np.prod(vcoord_shape) > 0.80:
        # Vcoord data is decreasing on interpolation axis, (at least 80% is)
        idx_gt = 1
        idx_lt = 0
    else:
        # Data is increasing on interpolation axis
        idx_gt = 0
        idx_lt = 1

    if data.ndim >= vcoord.ndim:
        # Handle case where data has the same dimensions or data has more dimensions
        # compared to vcoord (e.g. vcoord is 4D, data is 4D, or vcoord is 1D, data is 4D)
        out_shape = list(data.shape)
    else:
        # Handle case where data has fewer dimensions than vcoord
        # (e.g. data is 1-D vcoord is N-D)
        out_shape = list(vcoord.shape)
    out_shape[1] = vlevels.shape[0]

    out_data = np.zeros(out_shape) + np.nan

    for lev_idx, lev in enumerate(vlevels):
        if idx_gt == 0:
            # Case where vcoord data is increasing, find index where vcoord below [:-1]
            # is equal or less than desired lev, and vcoord above [1:] is greater than
            # lev, this means <data> for lev is between these points, use
            # weight to determine exactly where
            idx = np.squeeze(np.where(np.logical_and(vcoord[:, :-1, ...] <= lev,
                                                     vcoord[:, 1:, ...] > lev)))
        else:
            # This does the same, but where vcoord is decreasing with index, so find
            # where vcoord below [:-1] is greater, and vcoord above [1:] is less or equal
            idx = np.squeeze(np.where(np.logical_and(vcoord[:, :-1, ...] > lev,
                                                     vcoord[:, 1:, ...] <= lev)))

        # Create copies of this index, so they can be modified for weighting functions
        # and output array
        idx_abve = idx.copy()
        idx_belw = idx.copy()
        out_idx = idx.copy()

        # The interpolation axis index (1) for output is the level index (lev_idx)
        out_idx[1, :] = lev_idx

        # Weighting function 'above' is index +1 for decreasing, or index +0 for decr.
        idx_abve[1, :] += idx_gt
        # Weighting function 'below' is index +0 for decreasing, or index +1 for decr.
        idx_belw[1, :] += idx_lt

        # Change indicies back into tuples so numpy.array.__getitem__ understands them
        idx_abve = tuple(idx_abve)
        idx_belw = tuple(idx_belw)
        out_idx = tuple(out_idx)

        # Weighting function for distance above lev
        wgt1 = ((lev - vcoord[idx_belw]) / (vcoord[idx_abve] - vcoord[idx_belw]))

        # Weighting function for distance below lev
        wgt0 = 1.0 - wgt1

        if data.ndim >= vcoord.ndim:
            # Handle case where data has same or more dimensions than vcoord
            out_data[out_idx] = (wgt0 * data[idx_belw] + wgt1 * data[idx_abve])
        else:
            # Handle case where data has fewer dimensions than vcoord
            out_data[out_idx] = (wgt0 * data[idx_belw[1]] + wgt1 * data[idx_abve[1]])

    return np.squeeze(out_data)


def convert_radians_latlon(lat, lon):
    """
    Convert input lat/lon array to radians if input is degrees, do nothing if radians.

    Parameters
    ----------
    lat : array_like
        ND array of latitude
    lon : array_like
        ND array of longitude

    Returns
    ----------
    lat : array_like
        ND array of latitude in radians
    lon : array_like
        ND array of longitude in radians
    """
    if (np.max(np.abs(lat)) - np.pi / 2.0) > 1.0:
        lat_out = lat * RAD
    else:
        lat_out = lat

    if(np.min(lon) < 0 and np.max(lon) > 0 and
       np.abs(np.max(np.abs(lon)) - np.pi) > np.pi):
        lon_out = lon * RAD
    elif np.abs(np.max(np.abs(lon)) - np.pi * 2) > np.pi:
        lon_out = lon * RAD
    else:
        lon_out = lon

    return lat_out, lon_out


def dlon_dlat(lon, lat, cyclic=True):
    """
    Calculate distance along lat/lon axes on spherical grid.

    Parameters
    ----------
    lat : array_like
        ND array of latitude
    lon : array_like
        ND array of longitude

    Returns
    ----------
    dlong : array_like
        NLat x Nlon array of horizontal distnaces along longitude axis
    dlatg : array_like
        NLat x Nlon array of horizontal distnaces along latitude axis
    """
    # Check that lat/lon are in radians
    lat, lon = convert_radians_latlon(lat, lon)

    # Calculate centre finite difference of lon / lat
    dlon = lon[2:] - lon[:-2]
    dlat = lat[2:] - lat[:-2]

    # If we want cyclic data, repeat dlon[0] and dlon[-1] at edges
    if cyclic:
        dlon = np.append(dlon[0], dlon)      # cyclic boundary in East
        dlon = np.append(dlon, dlon[-1])     # cyclic boundary in West
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = np.meshgrid(lon[1:-1], lat)

    dlat = np.append(lat[1] - lat[0], dlat)    # boundary in South
    dlat = np.append(dlat, lat[-1] - lat[-2])  # boundary in North
    dlong, dlatg = np.meshgrid(dlon, dlat)

    # Lon/Lat differences in spherical coords
    dlong = dlong * EARTH_R * np.cos(lat2d)
    dlatg = dlatg * EARTH_R

    return dlong, dlatg


def rel_vort(uwnd, vwnd, lat, lon, cyclic=True):
    """
    Calculate the relative vorticity given zonal (uwnd) and meridional (vwnd) winds.

    Parameters
    ----------
    uwnd : array_like
        Zonal wind >=2 dimensional (time, level, lat, lon)
    vwnd : array_like
        Meridional wind >=2 dimensional, same dimensions as uwnd
    lat : array_like
        Latitude array, 1 dimensional with lat.shape[0] == uwnd.shape[-2]
    lon : array_like
        Longitude array, 1 dimensional with lon.shape[0] == uwnd.shape[-1]
    cyclic : boolean
        Flag to indicate if data is cyclic in longitude direction

    Returns
    -------
    rel_vort : array_like
        Relative vorticity V = U x V (cross product of U and V wind)
        If cyclic, returns same dimensions as uwnd & vwnd, if not it is
        (*vwnd.shape[0:-1], vwnd.shape[-1] - 2)
    """
    # Check that lat/lon are in radians
    lat, lon = convert_radians_latlon(lat, lon)

    # Get dlon and dlat in spherical coords
    dlong, dlatg = dlon_dlat(lon, lat, cyclic)

    # Generate quasi-broadcasts of lat/lon differences for divisions
    if uwnd.ndim == 4:
        dlong = dlong[np.newaxis, np.newaxis, ...]
        dlatg = dlatg[np.newaxis, np.newaxis, ...]
    elif uwnd.ndim == 3:
        dlong = dlong[np.newaxis, ...]
        dlatg = dlatg[np.newaxis, ...]
    elif uwnd.ndim < 2:
        raise ValueError('Not enough dimensions '
                         'for relative vorticity: {}'.format(uwnd.shape))
    elif uwnd.ndim > 4:
        raise NotImplementedError('Too many dimensions for this method: {} not yet'
                                  'implemented'.format(uwnd.shape))

    dvwnd = diff_cfd(vwnd, axis=-1, cyclic=True)
    duwnd = diff_cfd(uwnd, axis=-2, cyclic=False)

    # Divide vwnd differences by longitude differences
    dvdlon = dvwnd / dlong
    # Divide uwnd differences by latitude differences
    if cyclic:
        dudlat = duwnd / dlatg
    else:
        # If data isn't cyclic, then d(vwnd) and d(lon) will be one dim shorter on last
        # dimension, so we need to make up for that in the d(uwnd)/d(lat) field to match
        dudlat = duwnd[..., 1:-1] / dlatg

    return dvdlon - dudlat


def dth_dp(theta_in, data_in):
    """
    Calculates vertical derivative on even (theta) levels.

    Parameters
    ----------
    theta_in : array_like, 1D
        Vertical coordinate (potential temperature)
    data_in : array_like, ND, where data_in.shape[1] == theta_in.shape[0]
        Data on theta levels

    Returns
    -------
    out_data : array_like ND, same as data
        Centered finite difference for data_in[:, 1:-1, ...], backward/forward for
        bottom/top boundaries
    """
    # Calculate centred finite differences for theta (1D) and data on theta lvls (>=2D)
    dth = theta_in[2:] - theta_in[:-2]
    ddata = data_in[:, 2:, ...] - data_in[:, :-2, ...]

    # Calculated backward finite difference for bottom and top layers
    dth = np.append(theta_in[1] - theta_in[0], dth)
    dth = np.append(dth, theta_in[-1] - theta_in[-2])
    ddata = np.append((data_in[:, 1, ...] - data_in[:, 0, ...])[:, np.newaxis, ...],
                      ddata, axis=1)
    ddata = np.append(ddata, (data_in[:, -1, ...] -
                              data_in[:, -2, ...])[:, np.newaxis, ...], axis=1)

    if data_in.ndim == 4:
        return dth[np.newaxis, :, np.newaxis, np.newaxis] / ddata
    elif data_in.ndim == 3:
        return dth[np.newaxis, :, np.newaxis] / ddata
    elif data_in.ndim == 2:
        return dth[np.newaxis, :] / ddata
    else:
        raise ValueError('Incorrect number of dimensons: {}'.format(data_in.shape))


def ipv(uwnd, vwnd, tair, pres, lat, lon, th_levels=TH_LEV):
    """
    This method calculates isentropic PV on theta surfaces
    Note: interpolation assumes pressure is monotonically increasing.

    Parameters
    ----------
    uwnd : array_like
        3 or 4-D zonal wind component (t, p, y, x) or (p, y, x)
    vwnd : array_like
        3 or 4-D meridional wind component (t, p, y, x) or (p, y, x)
    tair : array_like
        3 or 4-D air temperature (t, p, y, x) or (p, y, x)
    pres : array_like
        1D pressure in Pa
    lat : array_like
        1D latitude in degrees
    lon : array_like
        1D longitude in degrees
    th_levels : array_like
        1D Theta levels on which to calculate PV


    Returns
    -------
    ipv : array_like
        3 or 4-D isentropic potential vorticity in units
        of m-2 s-1 K kg-1 (e.g. 10^6 PVU)
    p_th : array_like
        Pressure on isentropic levels [Pa]
    u_th : array_like
        Zonal wind on isentropic levels [m/s]
    """

    # Calculate potential temperature on isobaric (pressure) levels
    thta = theta(tair, pres)
    # Interpolate zonal, meridional wind, pressure to isentropic from isobaric levels
    u_th = vinterp(uwnd, thta, th_levels)
    v_th = vinterp(vwnd, thta, th_levels)
    p_th = vinterp(pres, thta, th_levels)

    # Calculate IPV on theta levels
    ipv_out = ipv_theta(u_th, v_th, p_th, lat, lon, th_levels)

    return ipv_out, p_th, u_th


def ipv_theta(uwnd, vwnd, pres, lat, lon, th_levels):
    """
    This method calculates isentropic PV on theta surfaces from data on theta levels
    Note: interpolation assumes pressure is monotonically increasing.

    Parameters
    ----------
    uwnd : array_like
        3 or 4-D zonal wind component (t, p, y, x) or (p, y, x)
    vwnd : array_like
        3 or 4-D meridional wind component (t, p, y, x) or (p, y, x)
    pres : array_like
        3 or 4-D pressure in Pa (t, p, y, x) or (p, y, x)
    lat : array_like
        1D latitude in degrees
    lon : array_like
        1D longitude in degrees
    th_levels : array_like
        1D Theta levels on which to calculate PV


    Returns
    -------
    ipv : array_like
        3 or 4-D isentropic potential vorticity in units
        of m-2 s-1 K kg-1 (e.g. 10^6 PVU)
    """
    # Calculate relative vorticity on isentropic levels
    rel_v = rel_vort(uwnd, vwnd, lat, lon)

    # Calculate d{Theta} / d{pressure} on isentropic levels
    dthdp = 1.0 / diffz(pres, th_levels)

    # Calculate Coriolis force
    # First, get axis matching latitude to input data
    lat_bcast = [np.newaxis] * rel_v.ndim
    lat_axis = np.where(np.array(rel_v.shape) == lat.shape[0])[0][0]
    lat_bcast[lat_axis] = slice(None)
    f_cor = 2.0 * OM * np.sin(lat[lat_bcast] * RAD)

    # Calculate IPV, then correct for y-derivative problems at poles
    ipv_out = -GRV * (rel_v + f_cor) * dthdp
    for pole_idx in [0, -1]:
        # This sets all points in longitude direction to mean of all points at the pole
        ipv_out[..., pole_idx, :] = np.mean(ipv_out[..., pole_idx, :], axis=-1)[..., None]

    # Return isentropic potential vorticity
    return ipv_out


def theta(tair, pres):
    """
    Calculate potential temperature from temperature and pressure coordinate.

    Parameters
    ----------
    tair : array_like
        ND array of air temperature [in K]
    pres : array_like
        Either ND array of pressure same shape as `tair`, or 1D array of pressure where
        its shape is same as one dimension of `tair` [in Pa]

    Returns
    -------
    theta : array_like
        ND array of potential temperature in K, same shape as `tair` input
    """
    r_d = 287.0
    c_p = 1004.0
    kppa = r_d / c_p
    if np.max(pres) < 90000.0:
        p_0 = 1000.0
    else:
        p_0 = 100000.0  # Don't be stupid, make sure pres and p_0 are in the same units!

    if tair.ndim == pres.ndim:
        p_axis = pres
    else:
        # Find which coordinate of tair is same shape as pres
        zaxis = tair.shape.index(pres.shape[0])

        # Generates a list of [None, None, ..., None] whose length is the number of
        # dimensions of tair, then set the z-axis element in this list to be a slice
        # of None:None This accomplishes same thing as [None, :, None, None] for
        # ndim=4, zaxis=1
        slice_idx = [None] * tair.ndim
        slice_idx[zaxis] = slice(None)

        # Create an array of pres so that its shape is (1, NPRES, 1, 1) if zaxis=1, ndim=4
        p_axis = pres[slice_idx]

    return tair * (p_0 / p_axis) ** kppa


def inv_theta(theta, pres):
    """
    Calculate potential temperature from temperature and pressure coordinate.

    Parameters
    ----------
    theta : array_like
        Either ND array of potential temperature same shape as `pres`, or 1D array of
        potential temperature where its shape is same as one dimension of `pres` [in K]
    pres : array_like
        ND array of pressure [in Pa]

    Returns
    -------
    theta : array_like
        ND array of air temperature in K, same shape as `tair` input
    """
    r_d = 287.0
    c_p = 1004.0
    kppa = r_d / c_p
    if np.max(pres) < 90000.0:
        p_0 = 1000.0
    else:
        p_0 = 100000.0  # Don't be stupid, make sure pres and p_0 are in the same units!

    if theta.ndim == pres.ndim:
        th_axis = theta
    else:
        # Find which coordinate of tair is same shape as pres
        zaxis = pres.shape.index(theta.shape[0])

        # Generates a list of [None, None, ..., None] whose length is the number of
        # dimensions of tair, then set the z-axis element in this list to be a slice
        # of None:None This accomplishes same thing as [None, :, None, None] for
        # ndim=4, zaxis=1
        slice_idx = [None] * pres.ndim
        slice_idx[zaxis] = slice(None)

        # Create an array of pres so that its shape is (1, NPRES, 1, 1) if zaxis=1, ndim=4
        th_axis = theta[slice_idx]

    return th_axis * (p_0 / pres) ** -kppa
