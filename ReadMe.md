Information on how to run the code and how it works.
# Running the code
Use of the [Anaconda Python](https://www.anaconda.com/download/) distribution is reccomended, as is Python 3.6 or newer.
## Required Python modules

#### Required for diagnostic plots:
----

	basemap
	matplotlib

#### Required for running the jet metric:
---

	netCDF4
	numpy
	psutil
	PyYAML
	scipy


### Installing for Python 2.7
`conda install --file requirements_27.txt`

### Installing for Python 3+
The version of `basemap` available from Anaconda, `1.0.7`, is not compatable with Python >= 3. The `conda-forge` channel has `v1.1.0` available.

`conda install --file requirements_36.txt -c conda-forge`


## Data dependencies
---
Monthly data is recommended but daily data is an option.

Required fields if potential vorticity is not available:

* zonal wind (u)
* meridional wind (v)
* atmospheric temperature (T)

If isentropic potential vorticity is available, then on isentropic levels:

* zonal wind (u)
* potential vorticity (pv)


## How the code is structured and how it works:
---


The highest level code is `run_stj.py`. Within this file the following changes are required:

1. Two [YAML](http://www.yaml.org/start.html) configuration files are used
    - STJ config (for properties of jet finding metric)
    - Data config (for properties of input data.
2. The data configuration file is set within the STJ configuration file. Examples of both can be found in the `conf/` directory.
	1. The `stj_config_default.yml` file contains the following options:

        - `data_cfg`: Location of data config file
        - `freq`: Input data frequency
        - `zonal_opt`: Output zonal mean (if 'mean') or individual longitude positions (if != 'mean')
        - `method`: Jet metric to use. Included are **STJPV** and **STJUMax**
        log_file: Log file name and location. If "{}" is included within this string (e.g. `stj_find_{}.log`) the time (from `datetime.now()`) at which the finder was initialised will be put into the file name (e.g. `stj_find_2017-11-02_14-08-32.log`)
        - `pv_value`: Potential vorticity level on which potential temperature is interpolated to find the jet (if using **STJPV** metric)
        - `fit_deg`: Also for **STJPV** metric, use this degree (integer) polynomial to fit the potential temperature on the `pv_value` surface
        - `min_lat`: Minimum latitude boundary (equatorward) on which to perform interpolation
        - `update_pv`: If isentropic PV (IPV) file(s) exist already, re-create them if this is set to `True`. If not, use files that exist
        - `year_s`: Year to start jet finding (Jan 1 of this year)
        - `year_e`: Year to end jet finding (Dec 31 of this year)
        - Dates may also be set in `run_stj.main()` function
        - `poly`: Polynomial to use, one of 'cheby', 'legendre', or 'poly' for Chebyshev, Legendre, or polynomial fit respectively

    2. The `data_config_default.yml` file contains the following options
        - `path`: Absolute path of input data
        - `wpath`: If `path` is _not_ writeable, absolute path to directory where IPV data can be written
        - `short_name`: String name to call this dataset
        - `single_var_file`: Each variables has its own file (if True)
        - `single_year_file`: Each years has its own file (if True)
        - `file_paths`: Names (within `path`) of input / output files for atmospheric variables.
            - If `single_var_file` is True, then this has:
                - `uwnd` (input u wind)
                - `vwnd` (input v wind)
                - `tair` (input air temperature)
                - `ipv`  (_output_ isentropic PV)
            - Otherwise, `file_paths` is just
                - `all`: File where all variables are (u, v, t, [optionally IPV])
                - `ipv`: File where IPV is, if different from `all`
        - `lon`: Name within netCDF file of 'longitude' variable
        - `lat`: Name within netCDF file of 'latitude' variable
        - `lev`: Name within netCDF file of 'level' variable
        - `time`: Name within netCDF file of 'time' variable
        - `ztype`: Type of levels (pressure, potential temperature, etc.)
        - `pfac`: Multiply pressure by this (float) to get units of Pascals
        - **See comments within `data_config_default.yml` for further details**

## How the STJPV metric works
   1. The Experiment object function GetIPV is an interface between the data and the code

   2. within GetIPV the function ipv is called (calc_ipv.py). This function is where ipv is calculate. Within calc_ipv.py the Fortran code is called.

   3. the only choice made to date is the levels to calculate IPV on  - which is 300 to 500 in 5K increments in this case.

   4. Within calcl_ipv the Fortran code is called to do the interpolations (Mike how would you describe the interpolation type? eg spline, linear etc)

   5. u,v and p are each interpolated onto the same vertical grid as potential temperature (in this case 300-500 every 5K)

   6. calculate the relative vorticity where  Z=dVdx-dUdy, where dx is grid box longitude step, dy the grid box latitude step, dV is the finite difference in meridional wind and dU the finite difference in zonal wind.

   7. Calculate the derivative dThdp and Coriolis parameter f_cor.

   8. Calculate IPV where `IPV = -g * (rel_v + f_cor) * dThdp`

5. Save the thermal tropopause and IPV fields for faster runtime of code.

    After the `STJ_PV_main.py` code has been run and the IPV data has been outputted the code then calculates the metric for the STJ latitude and intensity. This is done within the file STJ_IPV_metric.py and the main function is STJ_IPV_metric.py.

    1. Prepare data:
        1. Open IPV, tropopause height and zonal wind data

        2. Mask IPV array to manage grid box with inf.

        3. define latitudes and theta to interpolate to, every 0.2 deg and every 1K between 310-400K.

        4. Convert tropopause height to theta levels

        5. Define an array that links a calendar month to a season

    2. STJ metric:
        1. Interpolate the zonal mean IPV to 0.2deg lat and 1.0K theta.

        2. For each lat, find the theta level closest to 2.0 IPV.

        3. Test for unique elements of lat for the 2PV line and remove any repeats. Test that latitude is increasing and remove any elements which are not. This removes points on the 2PV line that look like a Z in the vertical.

        4. Polynomial fit the 2.0PV line. The degree of the polynomial fit is 10. Lower values for the fit were tested but were not able to capture the shape of the 2PV line.

        5. Differentiate the 2.0 IPV line using both finite difference and the chebyshev polynomial fit. The finite difference does not do a good enough job and was used during testing only.
               The second derivative is also calculated but was used only for testing.
        x
        6. Interpolate the thermal tropopause and dynamic tropopause to every 0.2 in latitude.

        7. Find the intersection of the thermal tropopause and dynamic tropopause (2.0 PV line). Subtract the two tropopause arrays and the minimum value is the intersection.
               The intersection is called the crossing point and will be a equatorial cutoff latitude for the STJ.

        8. Isolate the peaks of the 2.0PV line derivative. The goal is to find the maximum slope of the 2.0 PV line and hence the turning points of the first derivative.
               Between the crossing latitude and the pole find the local minima (maxima is SH). Sort the peaks in increasing lat.

        9. For testing purposed only, identify if the peaks are near the estimated annual mean of the subtropical and eddy driven jets.

        10. In order to reduce the number of peaks and isolate the STJ, find the shear between the 2.0PV line and surface at each peak identified. The STJ latitude is then the peak with the maximum shear.

    3. At the STJ latitude find its intensity.
        1. Interpolate zonal mean zonal wind to 0.2 deg and 1K.

        2. Then at the jet latitude, find the maximum wind in the column. This also gives the STJ theta level.

6. Then plot each monthly to show uwind, tropopause definitions etc.

7. When each STJ metric has been found, seasonally separate the timeseries and plot the jet latitude.


