#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute terms related to Eddy Kinetic Energy."""
import numpy as np
import matplotlib.pyplot as plt
import STJ_PV.utils as utils

__author__ = "Michael Kelleher, Penelope Maher"


class Kinetic_Eddy_Energies:
    def __init__(self, uwnd, vwnd, cfg):
        self.data = {"uwnd": uwnd, "vwnd": vwnd}

        self.t = cfg["time"]
        self.y = cfg["lat"]
        self.x = cfg["lon"]
        self.z = cfg["lev"]
        self.cpt = {}

    def get_components(self, zonal=True, time=True):
        """Get time and zonal anomalies and means for each component."""
        comp_ids = {}
        if zonal:
            comp_ids["zonal"] = ("z", self.x)
        if time:
            comp_ids["time"] = ("t", self.t)

        for dvar in self.data:
            for comp_id in comp_ids:
                cid = "{}_{}".format(dvar, comp_ids[comp_id][0])
                dim = comp_ids[comp_id][1]
                _mean = self.data[dvar].mean(dim=dim)
                _anom = self.data[dvar] - _mean
                self.cpt[cid + "m"] = _mean
                self.cpt[cid + "a"] = _anom

    def calc_momentum_flux(self, integration_top=None, rh=None, tau=None):
        r"""
        Calculate meridional change in eddy momentum flux.

        Notes
        -----
        :math:`S = \nabla\cdot\overline{F} = \frac{-1}{\cos(\phi)}\frac{\partial}{\partial phi}(\cos^2(\phi) [\overline{u'v'}])`

        """
        lam, phi = utils.convert_radians_latlon(
            self.data["uwnd"][self.x], self.data["uwnd"][self.y]
        )
        ac_phi = utils.EARTH_R * phi.pipe(np.cos)

        _, dphi = utils.xr_dlon_dlat(
            self.data["uwnd"], vlon=self.x, vlat=self.y
        )

        uv_zm = (self.cpt["uwnd_za"] * self.cpt["vwnd_za"]).mean(dim=self.x)
        dphi /= utils.EARTH_R

        d_f = utils.diff_cfd_xr(
            -ac_phi * uv_zm * phi.pipe(np.cos), dim=self.y, cyclic=False
        )
        self.del_f = (1.0 / ac_phi) * (d_f / dphi)

        # Make sure it's in the right order:
        self.del_f = self.del_f.transpose(self.t, self.y)
