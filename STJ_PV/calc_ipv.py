import numpy as np
import pdb


#specify the range and increment over which to calculate IPV
th_levels_trop   = np.arange(300,501,5)
rad              = np.pi/180.0  # radians per degree
Om               = 7.292e-5  # Angular rotation rate of earth    [rad]
g                = 9.81      # Acceleration due to gravity       [m/s^2]
__author__ = "Michael Kelleher" 

def ipv(u, v, t, p, lat, lon):
    """
        This method calculates isentropic PV on theta surfaces
        ----------
        Parameters
        ----------
        u : array_like
                3 or 4-D zonal wind component (t, p, y, x) or (p, y, x)
        v : array_like
                3 or 4-D meridional wind component (t, p, y, x) or (p, y, x)
        t : array_like
                3 or 4-D air temperature (t, p, y, x) or (p, y, x)
        p : array_like
                1D pressure in Pa
        lat : array_like
                1D latitude in degrees
        lon : array_like
                1D longitude in degrees

        Note: interpolation assumes pressure is monotonically increasing.
    
        Returns
        -------
        epv : array_like
                3 or 4-D isentropic potential vorticity in units
                of m-2 s-1 K kg-1 (e.g. 10^6 PVU)
    """
    #external fortran code
    import epv
    import interp

    if len(u.shape) == 4:
        nt, nz, ny, nx = u.shape
        four_dim = True
    elif len(u.shape) == 3:
        nz, ny, nx = u.shape
        four_dim = False

    nth = th_levels_trop.shape[0]
    TH  = theta(t, p)
    if four_dim:
        u_th    = interp.interp4d(TH, u, th_levels_trop, nt, nz, ny, nx, nth)
        v_th    = interp.interp4d(TH, v, th_levels_trop, nt, nz, ny, nx, nth)
        p_th    = interp.interp1d(TH, p, th_levels_trop, nt, nz, ny, nx, nth)
        rel_v   = epv.rel_vort(u_th, v_th, lat*rad, lon*rad, nt, nth, ny, nx)
        dThdp   = epv.dthdp(th_levels_trop, p_th, nt, nth, ny, nx)
        f_cor   = 2.0*Om*np.sin(lat[None, None, :, None]*rad)
    else:
        u_th    = interp.interp3d(TH, u, th_levels_trop, nz, ny, nx, nth)
        v_th    = interp.interp3d(TH, v, th_levels_trop, nz, ny, nx, nth)
        p_th    = interp.interp1d3d(TH, p, th_levels_trop, nz, ny, nx, nth)
        rel_v   = epv.rel_vort3d(u_th, v_th, lat*rad, lon*rad, nth, ny, nx)
        dThdp   = epv.dthdp3d(th_levels_trop, p_th, nth, ny, nx)
        f_cor   = 2.0*Om*np.sin(lat[None, :, None]*rad)

    return -g*(rel_v+f_cor)*dThdp,p_th,u_th

def theta(t, p):
    """
        Calculate potential temperature from temperature and pressure coordinate
    """
    Rd = 287.0
    Cp = 1004.0
    K = Rd / Cp
    p0 = 100000.0  # Don't be stupid, make sure p and p0 are in the same units!
    zaxis = t.shape.index(p.shape[0])

    if len(t.shape) == len(p.shape):
        p_axis = p
    elif len(p.shape) == 1:
        if len(t.shape) == 4:     # Assume data is (T, Z, Y, X) or (T, Z, X, Y)
            if zaxis == 0:
                p_axis = p[:, None, None, None]
            elif zaxis == 1:
                p_axis = p[None, :, None, None]
            elif zaxis == 2:
                p_axis = p[None, None, :, None]
            elif zaxis == 3:
                p_axis = p[None, None, None, :]
            else:
                raise IndexError('Axis {} out of bounds {}'.format(zaxis,
                                                                   len(t.shape)))

        elif len(t.shape) == 3:    # Data is (Z, x1, x2), (x1, Z, x2) or (x1, x2, Z)
            if zaxis == 0:
                p_axis = p[:, None, None]
            elif zaxis == 1:
                p_axis = p[None, :, None]
            elif zaxis == 2:
                p_axis = p[None, None, :]
            else:
                raise IndexError('Axis {} out of bounds {}'.format(zaxis,
                                                                   len(t.shape)))

        elif len(t.shape) == 2:  # Assume data is (T, Z)
            if zaxis == 0:
                p_axis  = p[None, :]
            elif zaxis == 1:
                p_axis  = p[:, None]
            else:
                raise IndexError('Axis {} out of bounds {}'.format(zaxis,
                                                                   len(t.shape)))
        else:                       # Data isn't in an expected shape, fail.
            raise ValueError('Input T is not correct shape {}'.format(t.shape))
    else:
        raise ValueError('Input P is not correct shape {}'.format(p.shape))
    return t * (p0 / p_axis) ** K



