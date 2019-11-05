# -*- coding: utf-8 -*-
"""Utility functions not specific to subtropical jet finding."""
from __future__ import division
import numpy as np
import xarray as xr
import xarray.ufuncs as xu
from scipy import interpolate as interp

__author__ = "Penelope Maher, Michael Kelleher"

# Constants to be used within this file
# specify the range and increment over which to calculate IPV
TH_LEV = np.arange(300.0, 501.0, 5)
RAD = np.pi / 180.0     # radians per degree
OM = 7.292e-5           # Angular rotation rate of earth    [rad]
GRV = 9.81              # Acceleration due to GRVity        [m/s^2]
EARTH_R = 6.371e6       # Radius of earth                   [m]
R_D = 287.0             # Dry gas constant                  [J kg^-1 K^-1]
C_P = 1004.0            # Specific heat of dry air          [J kg^-1 K^-1]
KPPA = R_D / C_P        # Ratio of gas constants


class NDSlicer(object):
    """
    Create an n-dimensional slice list.

    Parameters
    ----------
    axis : integer
        Axis on which to apply the slice
    ndim : integer
        Total number of dimensions of array to be sliced
    start, stop, step : integer, optional
        Index of beginning, stop and step width of the slice [start:stop:step]
        default for each is None.

    Examples
    --------
    Create random array, slice it::

        x = np.random.randn(5, 3)

        # Create slicer equivalent to [1:-1, :]
        slc = NDSlicer(0, x.ndim)
        print(x)
        [[ 0.68470539  0.87880216 -0.45086367]
         [ 1.06804045  0.63094676 -0.76633033]
         [-1.69841915  0.35207064 -0.4582049 ]
         [-0.56431067  0.62833728 -0.04101542]
         [-0.02760744  2.02814338  0.13195714]]
        print(x[slc[1:-1]])
        [[ 1.06804045  0.63094676 -0.76633033]
         [-1.69841915  0.35207064 -0.4582049 ]
         [-0.56431067  0.62833728 -0.04101542]]

    """

    def __init__(self, axis, ndim, start=None, stop=None, step=None):
        """N-Dimensional slice class for numpy arrays."""
        self.axis = axis
        self.ndim = ndim
        self.start = start
        self.stop = stop
        self.step = step
        self.slicer = None
        self.__getitem__(slice(start, stop, step))

    def __getitem__(self, key):
        """Create an n-dimensional slice list."""
        if isinstance(key, slice):
            self.start = key.start
            self.stop = key.stop
            self.step = key.step
        elif isinstance(key, int):
            self.start = key
            self.stop = key + 1
            self.step = None

        self.slicer = [slice(None)] * self.ndim
        self.slicer[self.axis] = slice(self.start, self.stop, self.step)
        return tuple(self.slicer)

    def slice(self, start=None, stop=None, step=None):
        """
        Create an n-dimensional slice list.

        This is a legacy compatibility method, calls `self.__getitem__`.

        Parameters
        ----------
        axis : integer
            Axis on which to apply the slice
        ndim : integer
            Total number of dimensions of array to be sliced
        start, stop, step : integer, optional
            Index of beginning, stop and step width of the slice [start:stop:step]
            default for each is None.

        Returns
        -------
        slicer : list
            list of slices such that all data at other axes are kept, one axis is sliced

        """
        self.__getitem__(slice(start, stop, step))


def vinterp(data, vcoord, vlevels):
    r"""
    Perform linear vertical interpolation.

    Parameters
    ----------
    data : array_like (>= 2D)
        Array of data to be interpolated
    vcoord : array_like (>= 2D, where data.shape[1] == vcoord.shape[1])
        Array representing the vertical structure (height/pressure/PV/theta/etc.) of
        `data`
    vlevels : array_like (1D)
        Levels, in same units as vcoord, to interpolate to

    Returns
    -------
    out_data : array_like, (data.shape[0], vlevels.shape[0], \*data.shape[2:])
        Data on vlevels

    Examples
    --------
    **data** and **vcoord** of the same shape

    Given some u-wind data ``uwnd`` that is on (time, pressure, lat, lon), a potential
    temperature field ``theta`` on the same dimensions (t, p, lat, lon), and a
    few selected potential temperature levels to which u-wind will be interpolated

        >>> th_levs = [275, 300, 325, 350, 375]
        >>> print(uwnd.shape)
        (365, 17, 73, 144)
        >>> print(theta.shape)
        (365, 17, 73, 144)
        uwnd_theta = vinterp(uwnd, theta, th_levs)
        >>> print(uwnd_theta.shape)
        (365, 5, 73, 144)

    This results in u-wind on the requested theta surfaces

    **data** is 1D **vcoord** is 4D

    Given some potential temperature levels ``th_levs = [275, 300, 325, 350, 375]``
    and a 4D array of potential vorticity ipv on (time, theta, lat, lon), and
    several selected PV levels

        >>> pv_levs = [1.0, 2.0, 3.0]
        >>> print(th_levs.shape)
        (5, )
        >>> print(ipv.shape)
        (365, 5, 73, 144)
        >>> theta_pv = vinterp(th_levs, ipv, pv_levs)
        >>> print(theta_pv.shape)
        (365, 3, 73, 144)

    This gives potential temperature on potential vorticity surfaces

    **data** is 4D **vcoord** is 1D

    Given some pressure levels ``levs = [1000., 900., 850., 500.]``
    and a 4D array of potential vorticity epv on (time, pressure, lat, lon), and
    several selected pressure levels

        >>> pres_new = [950.0, 800.0, 700.0]
        >>> print(levs.shape)
        (4, )
        >>> print(epv.shape)
        (365, 4, 73, 144)
        >>> epv_intrp = vinterp(epv, levs, pres_new)
        >>> print(epv_interp.shape)
        (365, 3, 73, 144)

    This gives potential vorticity on new pressure surfaces.

    """
    if vcoord.ndim == 1 and data.ndim > 1:
        # This handles the case where vcoord is 1D and data is N-D
        v_dim = int(np.where(np.array(data.shape) == vcoord.shape[0])[0])
        # numpy.broadcast_to only works for the last axis of an array, swap our shape
        # around so that vertical dimension is last, broadcast vcoord to it, then swap
        # the axes back so vcoord.shape == data.shape
        data_shape = list(data.shape)
        data_shape[-1], data_shape[v_dim] = data_shape[v_dim], data_shape[-1]

        vcoord = np.broadcast_to(vcoord, data_shape)
        vcoord = np.swapaxes(vcoord, -1, v_dim)

    vcoord_shape = list(vcoord.shape)
    vcoord_shape.pop(1)
    valid = np.min([np.prod(vcoord_shape) - np.sum(np.isnan(vcoord[:, 0, ...])),
                    np.prod(vcoord_shape) - np.sum(np.isnan(vcoord[:, -1, ...]))])

    if np.sum(vcoord[:, 0, ...] > vcoord[:, -1, ...]) / valid > 0.80:
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


def inc_with_z(vcoord, levname):
    """
    Find what proportion of `vcoord` is increasing along the `levname` axis.

    Parameters
    ----------
    vcoord : :class:`xarray.DataArray`
        array of vertical coordinate to test
    levname : str
        String name of vertical coordinate variable along which to test

    Returns
    -------
    pctinc : float
        Percentage of valid points in vcoord which are increasing
        with increasing index

    """
    _sabove = {levname: slice(1, None)}
    _sbelow = {levname: slice(None, -1)}

    # number of valid points on the 0th surface
    nvalid = vcoord.isel(**{levname: 0}).notnull().sum().drop(levname)

    # Places where the vcoord value on the 0th surface is less than the -1th (top)
    n_incr = (vcoord.isel(**{levname: 0}) < vcoord.isel(**{levname: -1}))

    # Sum the number where v[0] < v[-1], divide by number of valid points
    pctinc = n_incr.sum(skipna=True) / nvalid

    return pctinc


def _xrvinterp_single(data, vcoord, lev, levname='lev'):
    r"""
    Perform linear interpolation along vertical axis on an :class:`xarray.DataArray`.

    Parameters
    ----------
    data : :class:`xarray.DataArray` (>= 2D)
        array of data to be interpolated
    vcoord : :class:`xarray.DataArray`
        array representing the vertical structure
        (height/pressure/PV/theta/etc.) of `data`. This should be >= 2D, and
        data[levname].shape == vcoord[levname].shape
    lev : float
        Level, in same units as vcoord, to interpolate to
    levname : string
        Name of the vertical level coordinate variable upon
        which to interpolate

    Returns
    -------
    out_data : array_like, (data.shape[0], 1, \*data.shape[2:])
        Data on `lev`

    """
    # Slice shortcut dicts, above looks like [1:], below looks like [:-1]
    _sabove = {levname: slice(1, None)}
    _sbelow = {levname: slice(None, -1)}

    if inc_with_z(vcoord, levname) <= 0.8:
        # Fewer than 80% of points are increasing with index, swap directions
        _sbelow, _sabove = _sabove, _sbelow

    # Original levels from vcoord, should match original levels from data
    _olevs = vcoord[levname]

    # Selection dictionary to map levelname to coordinates above / below
    _coord_above = {levname: _olevs.isel(**_sabove)}
    _coord_below = {levname: _olevs.isel(**_sbelow)}

    # xarray.DataArray of booleans, true where the level "below" is less than
    # or equal to the desired level, false everywhere else
    below = (vcoord.isel(**_sbelow) <= lev).drop(levname)

    # xarray.DataArray of booleans, true where the level "above" is
    # greater than the desired level, false everywhere else
    above = (vcoord.isel(**_sabove) > lev).drop(levname)

    # Combine, this means select all points where
    # vcoord[LEVEL - 1] <= lev && vcoord[LEVEL + 1] > lev
    idx = xu.logical_and(above, below)

    # This is vcoord[:, 1:, ...], wherever idx is true and NaN everywhere else
    ix_ab = xr.where(idx.assign_coords(**_coord_above),
                     vcoord.isel(**_sabove), np.nan)

    # This is vcoord[:, :-1, ...], wherever idx is true and NaN everywhere else
    ix_bl = xr.where(idx.assign_coords(**_coord_below),
                     vcoord.isel(**_sbelow), np.nan)

    # Drop the coordinate information for the index xarrays, otherwise the
    # maths won't work (+, -, /, *)
    ix_ab = ix_ab.drop(levname)
    ix_bl = ix_bl.drop(levname)

    # Compute linear interpolation weights
    wgt1 = ((lev - ix_bl) / (ix_ab - ix_bl))
    wgt0 = 1.0 - wgt1

    # Perform the linear interpolation, sum over all original levels, to
    # eliminate the vertical dimension and remove all the NaNs
    out = (wgt0 * data.isel(**_sbelow).drop(levname) +
           wgt1 * data.isel(**_sabove).drop(levname)).sum(dim=levname)

    return out


def xrvinterp(data, vcoord, vlevs, levname, newlevname):
    r"""
    Perform vertical interpolation for several levels for an :class:`xarray.DataArray`.

    Parameters
    ----------
    data :  :class:`xarray.DataArray` (>= 2D)
        array of data to be interpolated
    vcoord :  :class:`xarray.DataArray`
        array representing the vertical structure
        (height/pressure/PV/theta/etc.) of `data`. This should be >= 2D, and
        data[levname].shape == vcoord[levname].shape
    vlevs : array_like (1D)
        Levels, in same units as vcoord, to interpolate to
    levname : string
        Name of the vertical level coordinate variable upon
        which to interpolate
    newlevname : string
        Name of new vertical level coordinate variable

    Returns
    -------
    out_data : array_like, (data.shape[0], vlevs.shape[0], \*data.shape[2:])
        Data on vlevels

    Notes
    -----
    If the input vertical coordinate data is increasing / decreasing with
    height in different places (e.g. potential vorticity across hemispheres),
    mask data and compute each part separately then add them together
    (if appropriate) or combine them, or whatever you'd like
    to do with them. We're not the boss of you :)

    """
    # Use a list-comprehension to assemble all the vertical coordinates
    intp = [_xrvinterp_single(data, vcoord, lev, levname) for lev in vlevs]
    # Concatenate the length (vlevs.shape[0]) list of xarray.DataArrays
    # The concat_dims is default, and avoids weirdness when data rank is small
    # (e.g. data.ndim < vcoord.ndim). Assing vlevs to be the values
    # for the new coordinate
    intp = xr.concat(intp, dim=newlevname).assign_coords(**{newlevname: vlevs})

    # Transpose the data, so the new level dimension is where the old level
    # dimension used to be in the original data or vcoord xarray.DataArray
    # depending on which has higher rank
    if data.ndim > vcoord.ndim:
        _dims = list(data.dims)
    else:
        _dims = list(vcoord.dims)

    lix = _dims.index(levname)
    _dims[lix] = newlevname
    intp = intp.transpose(*_dims)

    # Use where to mask out values that are extrapolated
    intp = intp.where(intp <= data.max()).where(intp >= data.min())

    return intp


def interp_nd(lat, theta_in, data, lat_hr, theta_hr):
    """
    Perform interpolation on 2-dimensions on up to 4-dimensional numpy array.

    Parameters
    ----------
    lat : array_like
        One dimensional latitude coordinate array, matches a dimension of `data`
    theta_in : array_like
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
    theta_dim = np.where(np.array(data.shape) == theta_in.shape[0])[0]
    if data.ndim == 2:
        data_f = interp.interp2d(lat, theta_in, data, kind='cubic')
        data_interp = data_f(lat_hr, theta_hr)

    elif data.ndim == 3:

        out_shape = list(data.shape)
        out_shape[lat_dim] = lat_hr.shape[0]
        out_shape[theta_dim] = theta_hr.shape[0]

        data_interp = np.zeros(out_shape)

        #For contexts when you have a 3d array but 2 dimensions are the same.
        cmn_axis = np.where(np.logical_and(np.array(data.shape) != lat.shape[0],
                                           np.array(data.shape) != theta_in.shape[0]))[0]
        # cmn_axis = np.where(out_shape == np.array(data.shape))[0]
        for idx0 in range(data.shape[cmn_axis]):

            data_f = interp.interp2d(lat, theta_in, data.take(idx0, axis=cmn_axis),
                                     kind='cubic')

            slc = [slice(None)] * data_interp.ndim
            slc[cmn_axis] = idx0
            data_interp[slc] = data_f(lat_hr, theta_hr)

    elif data.ndim == 4:

        out_shape = list(data.shape)
        out_shape[lat_dim] = lat_hr.shape[0]
        out_shape[theta_dim] = theta_hr.shape[0]
        data_interp = np.zeros(out_shape)

        # For contexts when you have a 3d array but 2 dimensions are the same.
        cmn_axis = np.where(np.logical_and(np.array(data.shape) != lat.shape[0],
                                           np.array(data.shape) != theta_in.shape[0]))[0]

        # cmn_axis = np.where(out_shape == np.array(data.shape))[0][:]
        for idx0 in range(data.shape[cmn_axis[0]]):
            for idx1 in range(data.shape[cmn_axis[1]]):
                data_slice = data.take(idx1, axis=cmn_axis[1]).take(idx0,
                                                                    axis=cmn_axis[0])
                data_f = interp.interp2d(lat, theta_in, data_slice, kind='cubic')
                # slc says which axis to place interpolated array on, it's what changes
                # with the loops
                slc = [slice(None)] * data_interp.ndim
                slc[cmn_axis[0]] = idx0
                slc[cmn_axis[1]] = idx1
                data_interp[slc] = data_f(lat_hr, theta_hr)
    return data_interp


def xrtheta(tair, pvar='level'):
    """
    Calculate potential temperature from temperature and pressure coordinate.

    Parameters
    ----------
    tair : :class:`xarray.DataArray`
        ND array of air temperature [in K]
    pvar : string
        Pressure level coordinate name for tair

    Returns
    -------
    theta : :class:`xarray.DataArray`
        ND array of potential temperature in K, same shape
        as `tair` input

    """
    # Attempt to figure pressure units
    try:
        p_units = tair[pvar].attrs['units']
    except KeyError:
        p_units = None

    # Default assumption is that pressure is in Pascals
    p_0 = 100000.0
    if p_units in ['hPa', 'mb', 'mbar', 'millibar', 'millibars']:
        # if pressure is in hPa (or similar), fix p_0
        p_0 /= 100.

    # Compute and return theta
    return tair * (p_0 / tair[pvar]) ** KPPA


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
    p_0 = 100000.0  # Don't be stupid, make sure pres and p_0 are in Pa!

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

    return tair * (p_0 / p_axis) ** KPPA


def xr_inv_theta(thta, pvar='level'):
    """
    Calculate potential temperature from temperature and pressure coordinate.

    Parameters
    ----------
    thta : :class:`xarray.DataArray`
        ND array of potential temperature [in K]
    pvar : string
        Pressure level coordinate name for tair

    Returns
    -------
    tair : :class:`xarray.DataArray`
        ND array of air temperature in K, same shape as `thta` input

    """
    # Attempt to figure pressure units
    try:
        p_units = thta[pvar].attrs['units']
    except KeyError:
        p_units = None

    # Default assumption is that pressure is in Pascals
    p_0 = 100000.0
    if p_units in ['hPa', 'mb', 'millibar']:
        # if pressure is in hPa (or similar), fix p_0
        p_0 /= 100.

    # Compute and return theta
    return thta * (p_0 / thta[pvar]) ** -KPPA


def inv_theta(thta, pres):
    """
    Calculate potential temperature from temperature and pressure coordinate.

    Parameters
    ----------
    thta : array_like
        Either ND array of potential temperature same shape as `pres`, or 1D array of
        potential temperature where its shape is same as one dimension of `pres` [in K]
    pres : array_like
        ND array of pressure [in Pa]

    Returns
    -------
    tair : array_like
        ND array of air temperature in K, same shape as `tair` input

    """
    p_0 = 100000.0  # Don't be stupid, make sure pres and p_0 are in Pa!

    if thta.ndim == pres.ndim:
        th_axis = thta
    else:
        # Find which coordinate of tair is same shape as pres
        zaxis = pres.shape.index(thta.shape[0])

        # Generates a list of [None, None, ..., None] whose length is the number of
        # dimensions of tair, then set the z-axis element in this list to be a slice
        # of None:None This accomplishes same thing as [None, :, None, None] for
        # ndim=4, zaxis=1
        slice_idx = [None] * pres.ndim
        slice_idx[zaxis] = slice(None)

        # Create an array of pres so that its shape is (1, NPRES, 1, 1) if zaxis=1, ndim=4
        th_axis = thta[slice_idx]

    return th_axis * (p_0 / pres) ** -KPPA


def lapse_rate(t_air, pres, vaxis=None):
    """
    Calculate the lapse rate of temperature in K/km from isobaric temperature data.

    Parameters
    ----------
    t_air : array_like
        N-D array of temperature on isobaric surfaces in K
    pres : array_like
        1-D array of pressure levels, matching one dimension of t_air, in hPa

    Returns
    -------
    dtdz : array_like
        N-D array of lapse rate in K/km matching dimensionality of t_air
    d_z : array_like
        N-D array of height differences in km between levels, same shape as t_air

    """
    # Common axis is vertical axis
    if pres.ndim == 1 or vaxis is None:
        ax_com = t_air.shape.index(pres.shape[0])
    elif pres.ndim == t_air.ndim:
        ax_com = vaxis
    else:
        raise ValueError('Dimensions do not match: T: {} P: {}'
                         .format(t_air.ndim, pres.ndim))

    # Create slices to use that are correct shape
    slc_t = NDSlicer(ax_com, t_air.ndim)
    if pres.ndim == t_air.ndim:
        slc_p = NDSlicer(ax_com, pres.ndim)
        bcast_nd = [slice(None)] * pres.ndim
    else:
        slc_p = NDSlicer(0, pres.ndim)
        # This generates a list of length ndim of t_air, (if 4-D then
        # [None, None, None, None]
        bcast_nd = [None] * t_air.ndim
        # This makes the common axis (vertical) a slice, if ax_com = 1 then it is the
        # same as saying pres[None, :, None, None], but allowing ax_com to be automagic
        bcast_nd[ax_com] = slice(None)

    # Calculate lapse rate in K/km
    d_p = (pres[slc_p.slice(1, None)] - pres[slc_p.slice(None, -1)])  # Units = Pa or hPa

    if np.max(pres) < 90000.0:
        d_p *= 100.0    # Now units should be in Pa
        pres_fac = 100.0
    else:
        pres_fac = 1.0

    # rho = p / (Rd * T)
    # Hydrostatic approximation dz = -dp/(rho * g)
    d_z = -d_p[bcast_nd] / (((pres[bcast_nd] * pres_fac) /
                             (R_D * t_air))[slc_t.slice(1, None)] * GRV) / 1000.0

    # Lapse rate [K/km] (-dt / dz)
    dtdz = -(t_air[slc_t.slice(1, None)] - t_air[slc_t.slice(None, -1)]) / d_z
    return dtdz, d_z


def trop_lev_1d(dtdz, d_z, thr=2.0, return_idx=False):
    """
    Given 1D arrays for lapse rate and change in height, and a threshold, find tropopause.

    Parameters
    ----------
    dtdz : array_like
        1D array of lapse rate (in same units as `thr`, usually K km^-1
    d_z : array_like
        1D array of change of height from previous level, same shape as `dtdz`, in same
        units as `dtdz` denominator units
    thr : float
        Lapse rate threshold for definition of tropopause, WMO/default is 2.0 K km^-1
    return_idx : bool
        Flag to return index of tropopause level. Default is False, if True output is
        `out_mask`, `idx` if False, output is `out_mask`

    Returns
    -------
    out_mask : array_like
        1D array, same shape as `dtdz` and `d_z`, of booleans, `True` everywhere except
        the tropopause level

    """
    lt_thr = dtdz < thr
    lt_transition = np.append(False, np.logical_and(np.logical_not(lt_thr[:-1]),
                                                    lt_thr[1:]))

    start_idx = np.where(np.logical_and(lt_thr, lt_transition))[0]
    out_mask = np.array([True] * d_z.shape[0], dtype=bool)

    for idx in start_idx:
        try:
            max_in_2km = dtdz[idx:idx + abs(np.cumsum(d_z[idx:]) - 2.0).argmin()].max()
        except ValueError:
            continue

        if max_in_2km <= thr:
            out_mask[idx] = False

    if return_idx:
        return out_mask, idx
    else:
        return out_mask


def find_tropopause_mask(dtdz, d_z, thr=2.0):
    """
    Use Reichler et al. 2003 method to calculate tropopause level.

    Parameters
    ----------
    dtdz : array_like
        1-D array of lapse rate in K/km
    d_z : array_like
        1-D array (same shape as `dtdz`) of difference in height between levels
    thr : float
        Lapse rate threshold in K/km (WMO definition is 2.0 K/km)

    Returns
    -------
    trop_level : integer
        Index of tropopause level on the vertical axis

    """
    # For now, we'll assume if 1-D data: vertical is dim 0, >= 2-D vertical is dim 1
    # Loops are slow, but still can't figure out a way beyond them :-(
    trop_level = np.empty(dtdz.shape, dtype=bool)
    if dtdz.ndim == 1:
        return trop_lev_1d(dtdz, d_z, thr)

    elif dtdz.ndim == 2:
        for ixt in range(dtdz.shape[0]):
            trop_level[ixt, :] = trop_lev_1d(dtdz[ixt, :], d_z[ixt, :], thr)

    elif dtdz.ndim == 3:
        for ixt in range(dtdz.shape[0]):
            for ixy in range(dtdz.shape[2]):
                trop_level[ixt, :, ixy] = trop_lev_1d(dtdz[ixt, :, ixy], d_z[ixt, :, ixy],
                                                      thr)
    elif dtdz.ndim == 4:
        for ixt in range(dtdz.shape[0]):
            for ixy in range(dtdz.shape[2]):
                for ixx in range(dtdz.shape[3]):
                    trop_level[ixt, :, ixy, ixx] = trop_lev_1d(dtdz[ixt, :, ixy, ixx],
                                                               d_z[ixt, :, ixy, ixx], thr)
    return trop_level


def get_tropopause(t_air, pres, thr=2.0, vaxis=1):
    """
    Return the tropopause temperature and pressure for WMO tropopause.

    Parameters
    ----------
    t_air : array_like
        ND array of temperature, where axis 1 is vertical axis
    pres : array_like
        ND array of pressure levels, shape is same as `t_air`
    thr : float
        Lapse rate threshold, default/WMO definition is 2.0 K km^-1

    Returns
    -------
    trop_temp, trop_pres : array_like
        Temperature and pressure at tropopause level, in (N-1)-D arrays, where dimension
        dropped is vertical axis, same units as input t_air and pres respectively

    """
    # Calculate the lapse rate, gives back lapse rate and d(height)
    dtdz, d_z = lapse_rate(t_air, pres, vaxis=vaxis)

    # Create tropopause level mask, use only the half levels (every other starting at 1)
    trop_level_mask = find_tropopause_mask(dtdz[:, 1::2, ...], d_z[:, 1::2, ...], thr=thr)

    # To get the tropopause temp/pres, mask the 4D arrays (at every other level)
    # then take the mean across level axis (now only one unmasked point) to give 3D data
    trop_temp = np.mean(np.ma.masked_where(trop_level_mask, t_air[:, 1::2, ...]), axis=1)
    trop_pres = np.mean(np.ma.masked_where(trop_level_mask, pres[:, 1::2, ...]), axis=1)
    return trop_temp, trop_pres


def get_tropopause_pres(t_air, pres, thr=2.0):
    """
    Return the tropopause temperature and pressure for WMO tropopause.

    Parameters
    ----------
    t_air : array_like
        ND array of temperature, where axis 1 is vertical axis
    pres : array_like
        1D array of pressure levels, shape is same as `t_air`.shape[1]
    thr : float
        Lapse rate threshold, default/WMO definition is 2.0 K km^-1

    Returns
    -------
    trop_temp, trop_pres : array_like
        Temperature and pressure at tropopause level, in (N-1)-D arrays, where dimension
        dropped is vertical axis, same units as input t_air and pres respectively

    """
    # Find half pressure levels
    pres_hf = (pres[:-1] - pres[1:]) / 2.0 + pres[1:]

    # Create full pressure array to interpolate temperature to
    pres_full = np.zeros(pres.shape[0] + pres_hf.shape[0])
    pres_full[::2] = pres
    pres_full[1::2] = pres_hf

    # Interpolate temperature to half pressure levels
    t_interp = interp.interp1d(pres, t_air, axis=1, kind='linear')(pres_full)

    # Broadcast pres_full to 4D, but pressure axis has to be last axis for broadcast_to
    # so, use temp shape, where index 1 and -1 are switched
    temp_shape = list(t_interp.shape)
    temp_shape[1], temp_shape[-1] = temp_shape[-1], temp_shape[1]
    pres_full_4d = np.swapaxes(np.broadcast_to(pres_full, temp_shape), -1, 1)

    return get_tropopause(t_interp, pres_full_4d, thr=thr, vaxis=1)


def get_tropopause_theta(theta_in, pres, thr=2.0):
    """
    Return the tropopause temperature and pressure for WMO tropopause.

    Parameters
    ----------
    theta_in : array_like
        1D array of potential temperature
    pres : array_like
        ND array of pressure levels, with one axis matching shape of theta_in
    thr : float
        Lapse rate threshold, default/WMO definition is 2.0 K km^-1

    Returns
    -------
    trop_temp, trop_pres : array_like
        Temperature and pressure at tropopause level, in (N-1)-D arrays, where dimension
        dropped is vertical axis, same units as input t_air and pres respectively

    """
    t_air = inv_theta(theta_in, pres)
    pres_levs = np.logspace(5, 3, theta_in.shape[0])
    t_pres = vinterp(t_air, pres, pres_levs)

    return get_tropopause_pres(t_pres, pres_levs, thr=thr)


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
    diff = data[slc[2:None]] - data[slc[None:-2]]

    if cyclic:
        # Cyclic boundary in "East"
        # Equiv to diff[..., 0] = data[..., 1:2] - data[..., -1:]
        d_1 = (data[slc[1:2]] - data[slc[-1:None]])
        diff = np.append(d_1, diff, axis=axis)

        # Cyclic boundary in "West"
        # Equiv to diff[..., -1] = data[..., 0:1] - data[..., -2:-1]
        diff = np.append(diff, (data[slc[0:1]] - data[slc[-2:-1]]), axis=axis)
    else:
        # Otherwise edges are forward/backward differences
        # Boundary in "South", (data[..., 1:2] - data[..., 0:1])
        diff = np.append((data[slc[1:2]] - data[slc[0:1]]), diff, axis=axis)

        # Boundary in "North" (data[..., -1:] - data[..., -2:-1])
        diff = np.append(diff, (data[slc[-1:None]] - data[slc[-2:-1]]), axis=axis)

    return diff


def diff_cfd_xr(data, dim='lon', cyclic=False):
    """
    Calculate centered finite difference along an axis with even spacing.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        N-D array of data of which to calculate the differences
    dim : string
        Dimension name of `data` along which differences are calculated
    cyclic : bool
        Flag to indicate whether `data` is cyclic on `axis`

    Returns
    -------
    diff : :class:`xarray.DataArray`
        N-D array of central finite differences
        of `data` along `axis`

    """
    # For centred differences along longitude direction (e.g.) this is
    # eqivalent to: diff = data[..., 2:] - data[..., :-2] for axis == -1
    diff = (data.isel(**{dim: slice(2, None)}).drop(dim)
            - data.isel(**{dim: slice(None, -2)}).drop(dim))
    diff = diff.assign_coords(**{dim: data[dim].isel(**{dim: slice(1, -1)})})

    if cyclic:
        # Cyclic boundary in "East"
        # Equiv to diff[..., 0] = data[..., 1:2] - data[..., -1:]
        d_1 = (data.isel(**{dim: 1}).drop(dim) -
               data.isel(**{dim: -1})).drop(dim)

        # Cyclic boundary in "West"
        # Equiv to diff[..., -1] = data[..., 0:1] - data[..., -2:-1]
        d_2 = (data.isel(**{dim: 0}).drop(dim) -
               data.isel(**{dim: -2})).drop(dim)

    else:
        # Otherwise edges are forward/backward differences
        # Boundary in "South", (data[..., 1:2] - data[..., 0:1])
        d_1 = data.isel(**{dim: 1}) - data.isel(**{dim: 0})
        d_2 = data.isel(**{dim: -1}) - data.isel(**{dim: -2})

    # Assign the coordiate to each "point"
    d_1 = d_1.assign_coords(**{dim: data[dim].isel(**{dim: 0})})
    d_2 = d_2.assign_coords(**{dim: data[dim].isel(**{dim: -1})})

    # Concatinate along the difference dimension
    diff = xr.concat((d_1, diff, d_2), dim=dim)

    # Use .transpose to make sure diff dims are same as data dims
    return diff.transpose(*data.dims)


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
        try:
            axis = data.shape.index(vcoord.shape[0])
        except ValueError:
            axis = vcoord.shape.index(data.shape[0])

    # Create array to hold vertical derivative
    dxdz = np.zeros(data.shape)

    # Create an n-dimensional broadcast along matching axis, same as [None, :, None, None]
    # for axis=1, ndim=4
    if vcoord.ndim < data.ndim:
        bcast = [np.newaxis] * data.ndim
        bcast[axis] = slice(None)
        d_z = (vcoord[1:] - vcoord[:-1])
        d_z2 = d_z[:-1][bcast]
        d_z1 = d_z[1:][bcast]
        # Create n-dimensional slicer along matching axis
        slc = NDSlicer(axis, data.ndim)
    else:
        slc_vc = NDSlicer(axis, vcoord.ndim)
        d_z = (vcoord[slc_vc[1:None]] - vcoord[slc_vc[None:-1]])
        d_z2 = d_z[slc_vc[None:-1]]
        d_z1 = d_z[slc_vc[1:None]]
        slc = NDSlicer(0, data.ndim)

    dxdz[slc[1:-1]] = ((d_z2 * data[slc[2:None]] + (d_z1 - d_z2) * data[slc[1:-1]] -
                        d_z1 * data[slc[None:-2]]) / (2.0 * d_z1 * d_z2))

    # Do forward difference at 0th level [:, 1, :, :] - [:, 0, :, :]
    dz1 = vcoord[1] - vcoord[0]
    dxdz[slc[0:1]] = (data[slc[0:1]] - data[slc[1:2]]) / dz1

    # Do backward difference at Nth level
    dz1 = vcoord[-1] - vcoord[-2]
    dxdz[slc[-1:None]] = (data[slc[-1:None]] - data[slc[-2:-1]]) / dz1

    return dxdz


def xrdiffz(data, vcoord, dim='lev'):
    """
    Calculate vertical derivative for data on uneven vertical levels.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        N-D array of input data to be differentiated, where
        data.shape[axis] == vcoord.shape[0]
    vcoord : :class:`xarray.DataArray`
        Vertical coordinate, 1D
    dim : string
        Vertical dimension name, must exist in both `data` and `vcoord`

    Returns
    -------
    dxdz : :class:`xarray.DataArray`
        N-D array of d(data)/d(vcoord), same shape as input `data`

    """
    d_z = (vcoord.isel(**{dim: slice(1, None)}).drop(dim) -
           vcoord.isel(**{dim: slice(None, -1)}).drop(dim))
    d_z1 = d_z.isel(**{dim: slice(1, None)})
    d_z2 = d_z.isel(**{dim: slice(None, -1)})

    # Central difference for uneven levels

    diff = ((d_z2 * data.isel(**{dim: slice(2, None)}).drop(dim) +
             (d_z1 - d_z2) * data.isel(**{dim: slice(1, -1)}).drop(dim) -
             d_z1 * data.isel(**{dim: slice(None, -2)}).drop(dim)) /
            (2.0 * d_z1 * d_z2))
    diff = diff.assign_coords(**{dim: data[dim].isel(**{dim: slice(1, -1)})})

    # Do forward difference at 0th level [:, 1, :, :] - [:, 0, :, :]
    dz1 = (vcoord.isel(**{dim: 1}).drop(dim) -
           vcoord.isel(**{dim: 0}).drop(dim))
    diff_0 = (data.isel(**{dim: 1}).drop(dim) -
              data.isel(**{dim: 0}).drop(dim)) / dz1
    diff_0 = diff_0.assign_coords(**{dim: data[dim].isel(**{dim: 0})})

    # Do backward difference at Nth level
    dz2 = (vcoord.isel(**{dim: -1}).drop(dim) -
           vcoord.isel(**{dim: -2}).drop(dim))
    diff_1 = (data.isel(**{dim: -1}).drop(dim) -
              data.isel(**{dim: -2}).drop(dim)) / dz2
    diff_1 = diff_1.assign_coords(**{dim: data[dim].isel(**{dim: -1})})

    # Combine all the differences
    diff = xr.concat((diff_0, diff, diff_1), dim=dim)

    return diff.transpose(*data.dims)


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


def xr_dlon_dlat(data, vlon='lon', vlat='lat', cyclic=True):
    """
    Calculate distance on lat/lon axes on spherical grid for :class:`xarray.DataArray`.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        DataArray with latitude and longitude coordinates
    vlon, vlat : string
        Variable names of latitude and longitude

    Returns
    ----------
    dlong : :class:`xarray.DataArray`
        NLat x Nlon array of horizontal distnaces
        along longitude axis in m
    dlatg : :class:`xarray.DataArray`
        NLat x Nlon array of horizontal distances
        along latitude axis in m

    """
    lon = data[vlon]
    lat = data[vlat]

    lon, lat = convert_radians_latlon(lon, lat)
    dlon = diff_cfd_xr(lon, dim=vlon, cyclic=False).compute()
    if cyclic:
        # If cyclic, the endpoints are central not fwd/bkw diffs, so assume
        # that the grid is regular, and double the edges
        dlon[0] *= 2
        dlon[-1] *= 2

    dlat = diff_cfd_xr(lat, dim=vlat, cyclic=False)

    dlon = dlon * EARTH_R * lat.pipe(np.cos)
    dlat = dlat * EARTH_R

    return dlon, dlat


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
        Relative vorticity V = U x V (cross product of U and V wind)
        If cyclic, returns same dimensions as uwnd & vwnd, if not it is
        ``(*vwnd.shape[0:-1], vwnd.shape[-1] - 2)``

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


def xr_rel_vort(uwnd, vwnd, dimvars, cyclic=True):
    r"""
    Calculate the relative vorticity given zonal (u) and meridional (v) winds.

    Parameters
    ----------
    uwnd : :class:`xarray.DataArray`
        Array of Zonal wind (with at least lat / lon dimensions)
    vwnd : :class:`xarray.DataArray`
        Array of Meridional wind with same dimensions as uwnd
    cyclic : boolean
        Flag to indicate if data is cyclic in longitude direction

    Returns
    -------
    rel_vort : :class:`xarray.DataArray`
        Relative vorticity V = U x V (cross product of U and V wind)
        with same dimensions as uwnd & vwnd

    """
    # Use .get to avoid key errors if dimvars is wrong, just defualt to
    # lat / lon, and if those are wrong, it'll fail later
    vlon = dimvars.get('lon', 'lon')
    vlat = dimvars.get('lat', 'lat')

    # Get dlon and dlat in spherical coords
    dlong, dlatg = xr_dlon_dlat(uwnd, vlon=vlon, vlat=vlat, cyclic=cyclic)

    dvwnd = diff_cfd_xr(vwnd, dim=vlon, cyclic=cyclic)
    duwnd = diff_cfd_xr(uwnd, dim=vlat, cyclic=False)

    # Divide vwnd differences by longitude differences
    dvdlon = dvwnd / dlong
    # Divide uwnd differences by latitude differences
    dudlat = duwnd / dlatg

    return dvdlon - dudlat


def dth_dp(theta_in, data_in):
    """
    Calculate vertical derivative on even (theta) levels.

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


def ipv(uwnd, vwnd, tair, pres, lat, lon, th_levels=None):
    """
    Calculate isentropic PV on theta surfaces.

    Notes
    -----
    Interpolation assumes pressure is monotonically increasing.

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
    th_levels : array_like, optional
        1D Theta levels on which to calculate PV. Defaults to 300K - 500K by 5K.


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
    if th_levels is None:
        th_levels = TH_LEV
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
    Calculate isentropic PV on theta surfaces from data on theta levels.

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


def xripv_theta(uwnd, vwnd, pres, dimvars):
    """
    Calculate isentropic PV on theta surfaces from data on theta levels.

    Parameters
    ----------
    uwnd : :class:`xarray.DataArray`
        3 or 4-D zonal wind component (t, theta, y, x) or (theta, y, x)
    vwnd : :class:`xarray.DataArray`
        3 or 4-D meridional wind component (t, theta, y, x) or (theta, y, x)
    pres : :class:`xarray.DataArray`
        3 or 4-D pressure in Pa (t, theta, y, x) or (theta, y, x)
    dimvars : dict
        Mapping of variable names for standard coordinates. This will default
        to 'theta' -> 'theta', 'lat' -> 'lat', 'lon' -> 'lon'

    Returns
    -------
    ipv : :class:`xarray.DataArray`
        3 or 4-D isentropic potential vorticity in units
        of m-2 s-1 K kg-1 (e.g. 10^6 PVU)

    """
    th_var = dimvars.get('lev', 'level')
    # Calculate relative vorticity on isentropic levels
    rel_v = xr_rel_vort(uwnd, vwnd, dimvars, cyclic=True)

    # Calculate d{Theta} / d{pressure} on isentropic levels
    dthdp = 1.0 / xrdiffz(pres, pres[th_var], dim=th_var)

    # Calculate Coriolis force
    # First, get axis matching latitude to input data
    f_cor = 2.0 * OM * (RAD * uwnd[dimvars['lat']]).pipe(np.sin)

    # Calculate IPV, then correct for y-derivative problems at poles
    ipv_out = -GRV * (rel_v + f_cor) * dthdp

    # Return isentropic potential vorticity
    return ipv_out


def xripv(uwnd, vwnd, tair, dimvars=None, th_levels=None):
    """
    Calculate isentropic PV on theta surfaces from :class:`xarray.DataArray`.

    Notes
    -----
    Interpolation assumes pressure is monotonically increasing.

    Parameters
    ----------
    uwnd : :class:`xarray.DataArray`
        3 or 4-D zonal wind component (t, p, y, x) or (p, y, x)
    vwnd : :class:`xarray.DataArray`
        3 or 4-D meridional wind component (t, p, y, x) or (p, y, x)
    tair : :class:`xarray.DataArray`
        3 or 4-D air temperature (t, p, y, x) or (p, y, x)
    dimvars : dict
        Mapping of variable names for standard coordinates. This will default
        to 'lev' -> 'level', 'lat' -> 'lat', 'lon' -> 'lon'
    th_levels : array_like, optional
        1D array of Theta levels on which to calculate PV.  Defaults to 300K - 500K by 5K.


    Returns
    -------
    ipv : :class:`xarray.DataArray`
        3 or 4-D isentropic potential vorticity in units
        of m-2 s-1 K kg-1 (e.g. 10^6 PVU)
    p_th : :class:`xarray.DataArray`
        Pressure on isentropic levels [Pa]
    u_th : :class:`xarray.DataArray`
        Zonal wind on isentropic levels [m/s]

    """
    if th_levels is None:
        th_levels = TH_LEV
    # import pdb;pdb.set_trace()
    th_levels = np.float32(th_levels)
    if dimvars is None:
        dimvars = {'lev': 'level', 'lat': 'lat', 'lon': 'lon'}

    vlev = dimvars['lev']

    # Calculate potential temperature on isobaric (pressure) levels
    thta = xrtheta(tair, pvar=vlev)

    # Interpolate zonal, meridional wind, pressure to isentropic from
    # isobaric levels

    u_th = xrvinterp(uwnd, thta, th_levels, levname=vlev,
                     newlevname=vlev)
    v_th = xrvinterp(vwnd, thta, th_levels, levname=vlev,
                     newlevname=vlev)

    # Check the units of uwnd.level to be sure to use Pa
    try:
        _punits = uwnd.level['units']
    except (KeyError, AttributeError):
        _punits = None
    if _punits in ['hPa', 'mb', 'millibar']:
        scale = 100.
    else:
        scale = 1.

    p_th = xrvinterp(scale * uwnd[vlev], thta, th_levels, levname=vlev,
                     newlevname=vlev)

    # Calculate IPV on theta levels
    ipv_out = xripv_theta(u_th, v_th, p_th, dimvars)

    return ipv_out, p_th, u_th
