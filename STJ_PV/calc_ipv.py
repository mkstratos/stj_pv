# -*- coding: utf-8 -*-
"""
Methods for calculating isentropic potential vorticity from data on pressure levels.
"""

import numpy as np


# specify the range and increment over which to calculate IPV
th_levels_trop = np.arange(300, 501, 5)
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

    if(np.sum(vcoord[:, 0, ...] > vcoord[:, -1, ...]) /
       np.prod([vcoord.shape[0], *vcoord.shape[2:]]) > 0.80):
        # Data is decreasing on interpolation axis, (at least 80% is)
        idx_gt = 1
        idx_lt = 0
    else:
        idx_gt = 0
        idx_lt = 1
    if data.ndim >= vcoord.ndim:
        out_data = np.zeros([data.shape[0], vlevels.shape[0], *data.shape[2:]]) + np.nan
    else:
        out_data = np.zeros([vcoord.shape[0], vlevels.shape[0],
                             *vcoord.shape[2:]]) + np.nan

    for lev_idx, lev in enumerate(vlevels):
        if idx_gt == 0:
            idx = np.squeeze(np.where(np.logical_and(vcoord[:, :-1, ...] <= lev,
                                                     vcoord[:, 1:, ...] > lev)))
        else:
            idx = np.squeeze(np.where(np.logical_and(vcoord[:, :-1, ...] > lev,
                                                     vcoord[:, 1:, ...] <= lev)))

        idx_abve = idx.copy()
        idx_belw = idx.copy()
        out_idx = idx.copy()

        out_idx[1, :] = lev_idx
        idx_abve[1, :] += (-idx_lt + 1)
        idx_belw[1, :] += (-idx_gt + 1)

        idx_abve = tuple(idx_abve)
        idx_belw = tuple(idx_belw)
        out_idx = tuple(out_idx)

        wgt1 = ((lev - vcoord[idx_belw]) / (vcoord[idx_abve] - vcoord[idx_belw]))
        wgt0 = 1.0 - wgt1
        if data.ndim >= vcoord.ndim:
            out_data[out_idx] = (wgt0 * data[idx_belw] + wgt1 * data[idx_abve])
        else:
            out_data[out_idx] = (wgt0 * data[idx_belw[1]] +
                                 wgt1 * data[idx_abve[1]])

    return out_data


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
    if np.max(np.abs(lat)) > np.pi / 2.0:
        lat = lat * RAD
    if np.max(np.abs(lon)) > np.pi * 2.0:
        lon = lon * RAD
    dlon = lon[2:] - lon[:-2]
    dlat = lat[2:] - lat[:-2]

    if cyclic:
        dlon = np.append(dlon[0], dlon)
        dlon = np.append(dlon, dlon[-1])
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = np.meshgrid(lon[1:-1], lat)

    dlat = np.append(lat[1] - lat[0], dlat)
    dlat = np.append(dlat, lat[-1] - lat[-2])
    dlon_nd, dlat_nd = np.meshgrid(dlon, dlat)

    dlon_nd = dlon_nd * EARTH_R * np.cos(lat2d)
    dlat_nd = dlat_nd * EARTH_R

    if uwnd.ndim == 4:
        dlon_nd = dlon_nd[np.newaxis, np.newaxis, ...]
        dlat_nd = dlat_nd[np.newaxis, np.newaxis, ...]
    elif uwnd.ndim == 3:
        dlon_nd = dlon_nd[np.newaxis, ...]
        dlat_nd = dlat_nd[np.newaxis, ...]
    elif uwnd.ndim < 2:
        raise ValueError('Not enough dimensions '
                         'for relative vorticity: {}'.format(uwnd.shape))
    elif uwnd.ndim > 4:
        raise NotImplementedError('Too many dimensions for this method: {} not yet'
                                  'implemented'.format(uwnd.shape))

    dvwnd = vwnd[..., 2:] - vwnd[..., :-2]
    if cyclic:
        dvwnd = np.append((vwnd[..., -1] - vwnd[..., 1])[..., np.newaxis], dvwnd, axis=-1)
        dvwnd = np.append(dvwnd, (vwnd[..., -2] - vwnd[..., 0])[..., np.newaxis], axis=-1)

    duwnd = uwnd[..., 2:, :] - uwnd[..., :-2, :]
    duwnd = np.append((uwnd[..., 1, :] - vwnd[..., 0, :])[..., np.newaxis, :],
                      duwnd, axis=-2)
    duwnd = np.append(duwnd, (uwnd[..., -1, :] - uwnd[..., -2, :])[..., np.newaxis, :],
                      axis=-2)

    dvdlon = dvwnd / dlon_nd
    if cyclic:
        dudlat = duwnd / dlat_nd
    else:
        dudlat = duwnd[..., 1:-1] / dlat_nd

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


def ipv(uwnd, vwnd, tair, pres, lat, lon):
    """
        This method calculates isentropic PV on theta surfaces
        ----------
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

        Note: interpolation assumes pressure is monotonically increasing.

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

    thta = theta(tair, pres)
    u_th = vinterp(uwnd, thta, th_levels_trop)
    v_th = vinterp(vwnd, thta, th_levels_trop)
    p_th = vinterp(pres, thta, th_levels_trop)
    rel_v = rel_vort(u_th, v_th, lat, lon)
    dthdp = dth_dp(th_levels_trop, p_th)
    f_cor = 2.0 * OM * np.sin(lat[np.newaxis, np.newaxis, :, np.newaxis] * RAD)

    return -GRV * (rel_v + f_cor) * dthdp, p_th, u_th


def theta(tair, pres):
    """
    Calculate potential temperature from temperature and pressure coordinate.
    """
    Rd = 287.0
    Cp = 1004.0
    K = Rd / Cp
    p0 = 100000.0  # Don't be stupid, make sure p and p0 are in the same units!
    zaxis = tair.shape.index(pres.shape[0])

    if len(tair.shape) == len(pres.shape):
        p_axis = pres
    elif len(pres.shape) == 1:
        if len(tair.shape) == 4:
            # Data is (Z, x1, x2, x3), (x0, Z, x2, x3), (x0, x1, Z, x3) or (x0, x1, x2, Z)
            if zaxis == 0:
                p_axis = pres[:, np.newaxis, np.newaxis, np.newaxis]
            elif zaxis == 1:
                p_axis = pres[np.newaxis, :, np.newaxis, np.newaxis]
            elif zaxis == 2:
                p_axis = pres[np.newaxis, np.newaxis, :, np.newaxis]
            elif zaxis == 3:
                p_axis = pres[np.newaxis, np.newaxis, np.newaxis, :]
            else:
                raise IndexError('Axis {} out of bounds {}'.format(zaxis,
                                                                   len(tair.shape)))

        elif len(tair.shape) == 3:
            # Data is (Z, x1, x2), (x0, Z, x2), or (x0, x1, Z)
            if zaxis == 0:
                p_axis = pres[:, np.newaxis, np.newaxis]
            elif zaxis == 1:
                p_axis = pres[np.newaxis, :, np.newaxis]
            elif zaxis == 2:
                p_axis = pres[np.newaxis, np.newaxis, :]
            else:
                raise IndexError('Axis {} out of bounds {}'.format(zaxis,
                                                                   len(tair.shape)))

        elif len(tair.shape) == 2:  # Assume data is (T, Z)
            # Data is (x0, Z), or (Z, x0)
            if zaxis == 0:
                p_axis = pres[np.newaxis, :]
            elif zaxis == 1:
                p_axis = pres[:, np.newaxis]
            else:
                raise IndexError('Axis {} out of bounds {}'.format(zaxis,
                                                                   len(tair.shape)))
        else:                       # Data isn't in an expected shape, fail.
            raise ValueError('Input T is not correct shape {}'.format(tair.shape))
    else:
        raise ValueError('Input P is not correct shape {}'.format(pres.shape))

    return tair * (p0 / p_axis) ** K
