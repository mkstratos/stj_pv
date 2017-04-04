"""
This module contains routines related to output of data to netCDF file.

Uses CF conventions to ensure standardized output
"""
from collections import OrderedDict
import numpy as np
import netCDF4 as nc


class NCOutVar(object):
    """
    This class contains the relavent information about an atmospheric variable.
    See the following site for information about the convention use:
    http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html

    Prototype variable properties dictionary (props)::

        props = {'name': 'VAR_STANDARD_NAME', 'scale_factor': SCALE,
                 'descr': 'VARIABLE DESCR', 'units': 'UNITS',
                 'short_name': 'output_name', 'calendar': 'standard',
                 'timevar': 'time', 'levvar': 'lev', 'latvar': 'lat', 'lonvar': 'lon',
                 'time_units': 'hours since 1900-01-01 00:00', 'lev_units': 'Pa',
                 'lon_units': 'degrees_east', 'lat_units': 'degrees_north'}

    Used in conjunction with writeTonetCDF() to make a standards
    compliant(-ish) output file

    Parameters
    ----------

    data_in : array_like
        N-Dimensional input data to be written to netCDF file
    props : dict
        Dictionary as described above, containing properties of the <data_in>
    coords : dict
        Dictionary containing at least one of ['time', 'lev', 'lat', 'lon'] or
        any combination of them, with associated coordinate variable
    """

    def __init__(self, data_in, props=None, coords=None):

        # Pull in required data, (data, diminsion sizes, its name,
        # description, units and netCDF output var)
        if props is None:
            self.gen_defualt_props()
        else:
            self.props = props

        # Coordinate variables are ordered (slow -> fast or t, p, y, x ), use OrderedDict
        # to make sure they stay that way
        self.coords = OrderedDict()
        self._set_coords(coords)

        self.data = data_in

    def _set_coords(self, coords_in):
        """
        Setup coordinate variables based on <self.props> and <self.coords> dictionaries
        """
        # For each possible coordinate variable, (time, lev, lat, lon)
        # set up the coordinate array with its shape and units
        for coord_type in ['time', 'lev', 'lat', 'lon']:
            if coord_type in coords_in:
                coord_var = self.props['{}var'.format(coord_type)]
                coord_name = coord_type
                coord_units = self.props['{}_units'.format(coord_type)]

                if coord_var == 'lev' and coord_units in ['Pa', 'kPa', 'hPa', 'mb']:
                    coord_name = 'air_pressure'
                elif coord_var == 'lev' and coord_units == 'K':
                    coord_name = 'air_potential_temperature'

                self.coords[coord_var] = {'cdata': coords_in[coord_type],
                                          'name': coord_name, 'units': coord_units}

    def gen_defualt_props(self):
        """
        Generate default set of variable properties.

        Used in case that properties are not passed to __init__ method, and props is None
        """
        props = {'name': 'VAR_STANDARD_NAME', 'scale_factor': 1.0,
                 'descr': 'VARIABLE DESCR', 'units': 'UNITS', 'short_name': 'DATA',
                 'calendar': 'standard', 'time_units': 'hours since 1900-01-01 00:00',
                 'timevar': 'time', 'levvar': 'lev', 'latvar': 'lat', 'lonvar': 'lon',
                 'lev_units': 'Pa', 'lon_units': 'degrees_east',
                 'lat_units': 'degrees_north'}
        self.props = props

    def get_props_from(self, copy_from):
        """
        Copy properties to another

        Parameters
        ----------
        copy_from : NCOutVar
            NCOutVar from which to get properties
        """
        self.props = dict(copy_from.props)
        self.coords = dict(copy_from.coords)
        self._set_coords()

    def set_prop(self, prop_name, prop_in=None):
        """
        Used to change a single property if prop name is string, or multiple if dict

        Parameters
        ----------
        prop_name : string
            Name of property to change
        prop_in : any
            Set self.props[prop_name] = prop_in
        """
        self.props[prop_name] = prop_in

    def set_props(self, prop_dict=None):
        """
        Used to set multiple properties at once

        Parameters
        ----------
        prop_dict: dict
            Format of {'prop_name': new prop value, ...}
        """
        for prop in prop_dict:
            self.set_prop(prop, prop_dict[prop])


def write_to_netcdf(data_in, out_file):
    """
    Write (a list of) NCOutVar variable(s) to a netCDF file.

    Uses zlib compression level 5.

    Parameters
    ----------
    data_in : list of NCOutVar
        List of NCOutVars to write to file
    out_file : string
        Name of file to write output
    """

    if not isinstance(data_in, list):
        data_in = [data_in]  # Just in case someone forgets to pass a list of variables

    # Open netCDF file for writing
    ncfile = nc.Dataset(out_file, mode='w')
    # Loop over coordinates, create those dimensions and variables in the netCDF file
    for coord_name in data_in[0].coords:
        if coord_name == 'time':
            dtype = np.dtype('double').char
            ncfile.createDimension(coord_name, size=None)
        else:
            dtype = np.dtype('float32').char
            ncfile.createDimension(coord_name,
                                   size=len(data_in[0].coords[coord_name]['cdata']))

        cvi = ncfile.createVariable(coord_name, dtype, (coord_name), zlib=True,
                                    complevel=5)

        if coord_name == 'time':
            cvi.calendar = data_in[0].props['calendar']

        cvi.units = data_in[0].coords[coord_name]['units']
        cvi.standard_name = data_in[0].coords[coord_name]['name']
        cvi[:] = data_in[0].coords[coord_name]['cdata']

    # Loop over output variables (must all be on same coordinates), write
    # data and attributes to file <out_file>
    for data in data_in:

        # Make sure our variable's short name isn't already in the file, if it is,
        # append _ to the name, until it's unique, but warn, because maybe this is bad
        if data.props['short_name'] in ncfile.variables.keys():
            print("WARNING: {} in file already".format(data.props['short_name']))
            while data.props['short_name'] in ncfile.variables.keys():
                data.props['short_name'] += "_"
            print("  USING: {}".format(data.props['short_name']))

        out_data = ncfile.createVariable(data.props['short_name'],
                                         np.dtype('float32').char,
                                         (list(data.coords.keys())), zlib=True,
                                         complevel=5)

        out_data.units = data.props['units']
        out_data.standard_name = data.props['name']
        out_data.description = data.props['descr']

        if 'scale_factor' in data.props:
            out_data.scale_factor = data.props['scale_factor']
        if 'offset' in data.props:
            out_data.add_offset = data.props['offset']
        if 'long_name' in data.props:
            out_data.long_name = data.props['long_name']
        out_data[:] = data.data

    ncfile.setncattr('Conventions', 'CF-1.6')
    ncfile.close()
