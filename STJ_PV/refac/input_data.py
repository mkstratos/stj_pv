"""Generate or load input data for STJ Metric."""
import pickle
import numpy as np
import netCDF4 as nc
# Dependent code
import calc_ipv
import data_out as dout

from thermal_tropopause import get_tropopause

__author__ = "Penelope Maher, Michael Kelleher"


class InputData(object):
    """
    Contains the relevant input data and routines for STJ Metric.

    Parameters
    ----------
    stj_props : run_stj.stj_props
        Object containing properties about the metric calculation to be done. Used to
        locate correct files, and variables within those files.
    """

    def __init__(self, stj_props):
        """Initialize InputData object, using stj_props object."""
        self.props = stj_props
        self.start_time = self.props.start_time
        self.end_time = self.props.end_time
        self.file_names = {}
        self.th_levels = stj_props.th_levels

        # Initialize attributes defined in open_files or open_ipv_data
        self.lon = None
        self.lat = None
        self.lev = None


class PresLevelData(InputData):

    def __init__(self, stj_props):
        # Call super-class' __init__ to set generic properties of InputData
        InputData.__init__(self, stj_props)

        for in_var in ['u', 'v', 't']:
            self.file_names[in_var] = '{}/{}'.format(self.props.in_files['in_dir'],
                                                     self.props.in_files[in_var])
        self.pres = None
        self.uwnd = None
        self.vwnd = None
        self.tair = None
        self.time = None
        self.time_units = None
        self.calendar = None

        # Initialize attributes defined on calc_ipv or open_ipv_data
        self.ipv = None
        self.p_th = None
        self.u_th = None

        # Initialize attributes defined on get_thermal_tropopause or open_ipv_data
        self.trop_h_temp = None
        self.trop_h_pres = None

        self.__get_data()
        self.ipv = np.ma.masked_invalid(self.ipv)

    def __get_data(self):
        """Get input data according to `run_flag` in stj_props."""
        if self.props.run_opts['run_flag'] == 'save':
            self.__open_input_files()
            self.calc_ipv()
            self.save_ipv()

        elif self.props.run_opts['run_flag'] in ['open', 'runnotsave']:
            self.__open_ipv_data()

    def __open_input_files(self):
        """
        Open U/V/T data to calculate IPV.

        Data assumed to be of the format [time, pressure, lat, lon].
        """
        for file_var in self.file_names:
            # Open file and extract data
            # in_var_list starts with getting coordinate names from stj_props object
            # then appends the netCDF variable name set in stj_props.var_names namedtuple
            in_var_list = list(self.props.coord_names.values())
            in_var_list.append(self.props.var_names[file_var].label)
            var = load_netcdf_file(self.file_names[file_var], in_var_list)
            lev_var = self.props.coord_names['lev']

            # If surface is not 0th element then flip data
            if var[lev_var].argmax() == 0:
                var[lev_var] = var[lev_var][::-1]
                var[file_var] = var[file_var][:, ::-1, :, :]

            # Add the data to the object
            setattr(self, file_var, var[file_var])

        # lon and lat assumed to be on the same grid for each variable, set these at end
        self.lon = var['lon']
        self.lat = var['lat']
        self.lev = var['lev']

        # check that input pressure is in Pascals
        if var[lev_var].max() < 90000.0:
            var[lev_var] = var[lev_var] * 100.0
        self.pres = var[lev_var]

        # restrict data in time from first desired time to last desired time. See function
        # PathFilenameERA
        self.tair = self.tair[self.start_time:self.end_time, ...]
        self.uwnd = self.uwnd[self.start_time:self.end_time, ...]
        self.vwnd = self.vwnd[self.start_time:self.end_time, ...]
        self.time = var['time'][self.start_time:self.end_time]
        self.time_units = var['time_units']
        self.calendar = var['calendar']

        self.props.log('Finished opening data')

    def calc_ipv(self):
        """Calculate isentropic potential vorticity from isobaric u, v, t."""
        self.props.log('Starting IPV calculation')
        # calculate IPV
        self.ipv, self.p_th, self.u_th = calc_ipv.ipv(self.uwnd, self.vwnd, self.tair,
                                                      self.pres, self.lat, self.lon,
                                                      self.th_levels)
        self.ipv *= 1e6  # Put PV in units of PVU
        self.props.log('Finished calculating IPV')

    def get_thermal_tropopause(self):
        """Calculate the tropopause height using the WMO thermal definition."""
        self.props.log('Start calculating tropopause height')

        # Get zonal mean temperature, levels should be in correct order on import
        t_zonal = np.nanmean(self.tair, axis=3)

        if self.lev[0] < self.lev[-1]:
            self.props.log('CHECK ON INPUT DATA, NOT IN sfc -> upper levels ORDER')
            self.lev = self.lev[::-1]
            t_zonal = t_zonal[:, ::-1, :]

        self.trop_h_temp, self.trop_h_pres = get_tropopause(t_zonal, self.lev)
        self.props.log('Finished calculating tropopause height')

    def save_ipv(self, out_file1=None, out_file2=None, output_type='.nc'):
        """
        Save IPV and tropopause data generated to a file, either netCDF4 or pickle.

        Parameters
        ----------
        out_file1 : string
            Output file name for pickle file, or first output file for netCDF4 file.
            Will contain ipv data and coords (plus tropopause data if `output_type` is
            pickle).
        out_file2 : string
            Output file name for tropopause data if `output_type` is netCDF4.
        output_type : string
            Either '.nc' if netCDF4 output is desired, or '.p' if pickle output desired.
        """
        if out_file1 is None:
            out_file1 = '{in_dir}{ipv1}'.format(**self.props.in_files)

        if out_file2 is None:
            out_file2 = '{in_dir}{ipv2}'.format(**self.props.in_files)

        self.props.log('Output IPV Data to:\n{}\n{}'.format(out_file1, out_file2))

        if output_type == '.p':
            output = {}

            output['time'] = self.time
            output['lev'] = self.lev
            output['lat'] = self.lat
            output['lon'] = self.lon

            output['ipv'] = self.ipv
            output['u_th'] = self.u_th

            output['trop_h_pres'] = self.trop_h_pres
            output['trop_h_temp'] = self.trop_h_temp

            pickle.dump(output, open('{}{}'.format(out_file1, output_type), 'wb'))

        else:

            ipv_out = dout.NCOutVar(self.ipv, coords={'time': self.time, 'lev': self.lev,
                                                      'lat': self.lat, 'lon': self.lon})

            ipv_out.set_props({'name': 'isentropic_potential_vorticity',
                               'descr': 'Potential vorticity on theta levels',
                               'units': 'PVU', 'short_name': 'ipv',
                               'levvar': 'theta_lev',
                               'time_units': self.time_units, 'calendar': self.calendar})

            dout.write_to_netcdf([ipv_out], '{}{}'.format(out_file1, output_type))

            u_th_out = dout.NCOutVar(self.u_th)
            u_th_out.get_props_from(ipv_out)
            u_th_out.set_props({'name': 'zonal_wind_component',
                                'descr': 'Zonal wind on isentropic levels',
                                'units': 'm s-1', 'short_name': 'u_th',
                                'levvar': 'theta_lev'})

            coords_2d = {'time': self.time, 'lat': self.lat}

            trop_h_pres_out = dout.NCOutVar(self.trop_h_pres, coords=coords_2d)
            trop_h_pres_out.set_props({'name': 'tropopause_level',
                                       'descr': 'Tropopause pressure level',
                                       'units': 'Pa', 'short_name': 'trop_h_pres',
                                       'time_units': self.time_units,
                                       'calendar': self.calendar})

            trop_h_temp_out = dout.NCOutVar(self.trop_h_temp, coords=coords_2d)
            trop_h_temp_out.get_props_from(trop_h_pres_out)
            trop_h_temp_out.set_props({'descr': 'Tropopause temperature',
                                       'short_name': 'trop_h_temp'})

            dout.write_to_netcdf([u_th_out, trop_h_pres_out, trop_h_temp_out],
                                 '{}{}'.format(out_file2, output_type))

        self.props.log('Finished writing\n{}\n{}'.format(out_file1, out_file2))

    def __open_ipv_data(self, filename_1=None, filename_2=None, file_type='.nc'):
        """
        Load variables from input files into the InputData object, setting as attributes.

        These input files are generated by `self.save_ipv()` method.

        Parameters
        ----------
        filename_1 : string
            Location of first file to read (if `filetype` == '.nc') or only file to read
            if `filetype` == '.p'.  Contains ipv and coordinate data.
        filename_2 : string
            Location of second file to read, contains zonal wind, and tropopause data.
        file_type : string
            Type of files to load, one of '.nc', '.p' for netCDF4 or pickle respectively.
        """
        if filename_1 is None:
            filename_1 = '{in_dir}{ipv1}'.format(**self.props.in_files)

        if filename_2 is None:
            filename_2 = '{in_dir}{ipv2}'.format(**self.props.in_files)

        if file_type == '.p':
            ipv_data = pickle.load(open('{}{}'.format(filename_1, file_type), 'rb'))
            for in_var in ipv_data:
                setattr(self, in_var, ipv_data[in_var])

        elif file_type == '.nc':
            var_list_1 = ['time', 'lev', 'lat', 'lon', 'ipv']
            ipv_data_1 = load_netcdf_file('{}{}'.format(filename_1, file_type),
                                          var_list_1)
            for in_var in ipv_data_1:
                setattr(self, in_var, ipv_data_1[in_var])

            var_list_2 = ['u_th', 'ipv_310', 'trop_h', 'trop_h_temp', 'trop_h_pres']
            ipv_data_2 = load_netcdf_file('{}{}'.format(filename_2, file_type),
                                          var_list_2)

            for in_var in ipv_data_2:
                setattr(self, in_var, ipv_data_2[in_var])


def load_netcdf_file(in_file, var_names):
    """
    Load a netCDF4 file into a dictionary, using a list of variable names.

    Parameters
    ----------
    in_file : string
        Location of file to be loaded
    var_names : list
        Names of variables to be loaded, including coordinate variables.
        e.g.: ['time', 'lev', 'lat', 'lon', 'ipv']

    Returns
    -------
    out_data : dict
        Dictionary of format {var_name : var_data, ...} for all vars specified in
        var_names
    """
    in_dset = nc.Dataset(in_file, 'r')
    out_data = {}
    for in_var in var_names:
        out_data[in_var] = in_dset.variables[in_var][:]
        if in_var == 'time':
            out_data['time_units'] = in_dset.variables[in_var].units
            out_data['calendar'] = in_dset.variables[in_var].calendar
    in_dset.close()

    return out_data
