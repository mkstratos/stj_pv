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

    return out_data


def rel_vort(uwnd, vwnd, lat, lon, cyclic=True):
    r"""
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
        Relative vorticity :math:`\zeta_h = \nabla \times \overrightarrow{V_h}`
        (cross product of gradient operator and  horizontal vector wind)
        If cyclic, returns same dimensions as uwnd & vwnd, if not it is::

            (*vwnd.shape[0:-1], vwnd.shape[-1] - 2)
    """

    # Check data for correct dimensionality
    if uwnd.shape != vwnd.shape or uwnd.ndim < 2 or vwnd.ndim < 2:
        raise ValueError('Incorrect dimensionality of u/v wind: U: {}, V: {}'
                         .format(uwnd.shape, vwnd.shape))
    elif uwnd.ndim > 4 or vwnd.ndim > 4:
        raise NotImplementedError('Too many dimensions for this method: U: {}, V: {}'
                                  .format(uwnd.shape, vwnd.shape))

    # Make sure lat and lon are in radians not degrees
    if np.max(np.abs(lat)) > np.pi / 2.0:
        lat = lat * RAD
    if np.max(np.abs(lon)) > np.pi * 2.0:
        lon = lon * RAD

    # Generate 2d lat/lon for weighting
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Calculate centred finite differences for lat/lon
    dlon = lon[2:] - lon[:-2]
    dlat = lat[2:] - lat[:-2]

    if cyclic:
        # If data are cyclic in longitude, add points to beginning/end of d{lon}
        dlon = np.append(dlon[0], dlon)
        dlon = np.append(dlon, dlon[-1])
    else:
        # Otherwise, calculate backward/forward difference at edges
        dlon = np.append(lon[1] - lon[0], dlon)
        dlon = np.append(dlon, lat[-1] - lat[-2])

    # Calculate backward/forward differences at top/bottom of domain
    dlat = np.append(lat[1] - lat[0], dlat)
    dlat = np.append(dlat, lat[-1] - lat[-2])

    # Make 2D (for now) mesh of d{lat} and d{lon}
    dlon_nd, dlat_nd = np.meshgrid(dlon, dlat)

    # Multiply each by appropriate factors for spherical geometry
    dlon_nd = dlon_nd * EARTH_R * np.cos(lat2d)
    dlat_nd = dlat_nd * EARTH_R

    # Make d{lon} and d{lat} have appropriate dimensionality to divide d{wind} by
    if uwnd.ndim == 4:
        dlon_nd = dlon_nd[np.newaxis, np.newaxis, ...]
        dlat_nd = dlat_nd[np.newaxis, np.newaxis, ...]
    elif uwnd.ndim == 3:
        dlon_nd = dlon_nd[np.newaxis, ...]
        dlat_nd = dlat_nd[np.newaxis, ...]
    # If uwnd.ndim == 2, d{lon} and d{lat} are already 2D

    # Calculate centred finite diff on vwind longitude axis
    dvwnd = vwnd[..., 2:] - vwnd[..., :-2]
    if cyclic:
        # If data is cyclic append centred difference across cyclic point
        dvwnd = np.append((vwnd[..., 1] - vwnd[..., -1])[..., np.newaxis], dvwnd, axis=-1)
        dvwnd = np.append(dvwnd, (vwnd[..., 0] - vwnd[..., -2])[..., np.newaxis], axis=-1)
    else:
        # Otherwise do backward/forward difference at edges
        dvwnd = np.append((vwnd[..., 1] - vwnd[..., 0])[..., np.newaxis], dvwnd,
                          axis=-1)
        dvwnd = np.append(dvwnd, (vwnd[..., -1] - vwnd[..., -2])[..., np.newaxis],
                          axis=-1)

    # Calculate centred finite difference on uwnd latitude axis
    duwnd = uwnd[..., 2:, :] - uwnd[..., :-2, :]

    # Append backward/forward difference at top/bottom of domain
    duwnd = np.append((uwnd[..., 1, :] - uwnd[..., 0, :])[..., np.newaxis, :], duwnd,
                      axis=-2)
    duwnd = np.append(duwnd, (uwnd[..., -1, :] - uwnd[..., -2, :])[..., np.newaxis, :],
                      axis=-2)

    # Return d{v}/d{lon} - d{u}/d{lat}
    return dvwnd / dlon_nd - duwnd / dlat_nd


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

    # Calculate relative vorticity on isentropic levels
    rel_v = rel_vort(u_th, v_th, lat, lon)

    # Calculate d{Theta} / d{pressure} on isentropic levels
    dthdp = dth_dp(th_levels, p_th)

    # Calculate Coriolis force
    f_cor = 2.0 * OM * np.sin(lat[np.newaxis, np.newaxis, :, np.newaxis] * RAD)

    # Calculate IPV, then correct for y-derivative problems at poles
    ipv_out = -GRV * (rel_v + f_cor) * dthdp
    for pole_idx in [0, -1]:
        # This sets all points in longitude direction to mean of all points at the pole
        ipv_out[..., pole_idx, :] = np.mean(ipv_out[..., pole_idx, :], axis=-1)[..., None]

    # Return isentropic potential vorticity, pressure on theta, u-wind on theta
    return ipv_out, p_th, u_th


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
