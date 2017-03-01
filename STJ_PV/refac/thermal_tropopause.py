"""Functions for finding thermal tropopause level."""
import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt
plt.style.use('ggplot')


__author__ = "Penelope Maher, Michael Kelleher"


def lapse_rate(t_air, pres):
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
    r_d = 287.0         # J K^-1 kg^-1  gas constant of dry air
    g = 9.81            # m/s^2

    # Common axis is vertical axis
    ax_com = np.squeeze(np.where(np.array(t_air.shape) == pres.shape[0])[0])

    # Create slices to use that are correct shape
    slc_p1 = [slice(None)] * t_air.ndim
    slc_m1 = [slice(None)] * t_air.ndim

    # Make slice plus one the same as [:, 1:, ...] if ax_com==1
    slc_p1[ax_com] = slice(1, None)
    # Make slice minus one the same as [:, :-1, ...] if ax_com==1
    slc_m1[ax_com] = slice(None, -1)

    # This generates a list of length ndim of t_air, (if 4-D then [None, None, None, None]
    slc_nd = [None] * t_air.ndim

    # This makes the common axis (vertical) a slice, if ax_com = 1 then it is the
    # same as saying pres[None, :, None, None], but allowing ax_com to be automagic
    slc_nd[ax_com] = slice(None)

    # Calculate lapse rate in K/km
    d_t = t_air[slc_p1] - t_air[slc_m1]      # Units = K
    d_p = (pres[1:] - pres[:-1]) * 100.0     # Units = Pa

    rho = (pres[slc_nd] * 100) / (r_d * t_air)  # rho = p / (Rd * T)
    d_z = -d_p[slc_nd] / (rho[slc_p1] * g) / 1000.0   # Hydrostatic approx.
    dtdz = (-d_t / d_z)                   # Lapse rate [K/km]
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


def get_tropopause(t_air, pres, thr=2.0):
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

    # Calculate the lapse rate, gives back lapse rate and d(height)
    dtdz, d_z = lapse_rate(t_interp, pres_full)

    # Create tropopause level mask, use only the half levels (every other starting at 1)
    trop_level_mask = find_tropopause_mask(dtdz[:, 1::2, ...], d_z[:, 1::2, ...], thr)

    # To get the tropopause temp/pres, mask the 4D arrays (at every other level)
    # then take the mean across level axis (now only one unmasked point) to give 3D data
    trop_temp = np.mean(np.ma.masked_where(trop_level_mask,
                                           t_interp[:, 1::2, ...]), axis=1)
    trop_pres = np.mean(np.ma.masked_where(trop_level_mask,
                                           pres_full_4d[:, 1::2, ...]), axis=1)
    return trop_temp, trop_pres
