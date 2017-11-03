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

1. The `run_stj.main()` function creates a `run_stj.JetFindRun` object, based on configuration parameters.

2. Start and end dates are set, and the `run_stj.JetFindRun.run()` method starts the run, where configuration files are checked
then the selected metric computes the jet postion in each hemisphere at each time.

3. If Isentropic PV input data does not exist, this is created and written as setup in the data configuration file

4. When using the **STJPV** metric the jet is identified in the following process:

    1. Interpolate to obtain potential temperature ($\Theta$) as a function of latitude on a surface
        of constant IPV, chosen in configuration file

    2. Numerically compute meridional gradient of this surface using a polynomal fit (Chebyshev polynomials of degree 8 used by default)

    3. The jet location is determined to be at a relative maximum in the northern hemisphere, or minimum
        in the southern hemisphere of the meridional gradient of potential temperature on the PV surface at each time and longitude

    4. If multiple extrema exist, the jet latitude has the largest zonal wind shear between the
        potential vorticity surface and the lowest available level (called the "surface")

    5. The zonal mean jet position for each time is then computed as the zonal median of the
        identified positions at all longitudes, ignoring those longitudes where no position is
        identified, if the `zonal_opt` is set to `"mean"` in the configuration, otherwise the position is output at each longitude
