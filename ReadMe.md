# STJ-PV
A subtropical jet finding framework, including the STJPV method introduced in

Maher, et al. (2019): *Is the subtropical jet shifting poleward?* *Climate Dynamics*.

[![DOI](https://img.shields.io/badge/DOI-10.11578/dc.20191106.1-orange/?style=flat-square&color=orange)](https://doi.org/10.1007/s00382-019-05084-6)


[![Documentation Status](https://readthedocs.org/projects/stj-pv/badge/?version=latest&style=flat-square)](https://stj-pv.readthedocs.io/en/latest/?badge=latest)
[![GitHub](https://img.shields.io/github/license/mkstratos/stj_pv?color=1E5AAF&label=License&style=flat-square&logo=Open%20Source%20Initiative)](https://github.com/mkstratos/stj_pv/blob/master/LICENSE)
[![Code DOI](https://img.shields.io/badge/Code_DOI_(v1.0.0)-10.11578%2Fdc.20191106.1-red/?style=flat-square&color=007833)](https://doi.org/10.11578/dc.20191106.1)


## Running the code
We reccomend:

* [Anaconda Python](https://www.anaconda.com/download/) distribution
* Python >= 3.6 (>= 3.5 required)
* Creating a new Anaconda environment (so package versions do not conflict between this and other projects)

### SETUP
Create a new Anaconda environment using:

`conda create -n stjpv python`

Activate your new environment with

`conda activate stjpv`

Then install the required packages as below.

### Installing STJ_PV
Clone (or [fork](https://github.com/mkstratos/stj_pv/fork)) this repository

`git clone git@github.com:mkstratos/stj_pv.git`

Enter the code's top level directory

`cd stj_pv`

Install the prerequisites
`conda install --file requirements.txt -c conda-forge`

**Note**: `basemap==1.0.7` available from Anaconda is not compatible with Python >= 3. Thus the `conda-forge` channel with `v1.1.0` must be used.

Install this module (STJ_PV) in development mode (`-e`)

`pip install -e .`

**Note the trailing "."** this will use the `setup.py` file to install this module, and allow it
to be imported using `import STJ_PV` or `from STJ_PV import run_stj` for example


## Testing with sample data
Enter the top-level code directory, and try the sample case:

---

    cd stj_pv/STJ_PV
    python run_stj.py --sample

This will output a file called: `NCEP_NCAR_DAILY_STJPV_pv2.0_fit6_y010.0_yN65.0_zmean_2009-01-23_2009-01-25.nc`
which has the latitude and theta position, and intensity in northern and southern hemispheres, each their own variable.


## Required Python modules

#### Required for running the jet metric:
---

	dask
	netCDF4
	numpy
	psutil
	PyYAML
	scipy
	xarray

#### Required for diagnostic plots:
----

    basemap
    matplotlib
    seaborn
    pandas

## Data dependencies
---
Monthly data is recommended but daily data is an option.

Required fields on isobaric levels if isentropic potential vorticity is not available:

* zonal wind (u)
* meridional wind (v)
* atmospheric temperature (T)

If isobaric potential vorticity is available, then on isobaric levels:

* zonal wind (u)
* atmospheric temperature (T)
* potential vorticity (pv)

If isentropic potential vorticity is available, then on isentropic levels:

* zonal wind (u)
* potential vorticity (pv)


## Code structure and organization:
---


The highest level code is `run_stj.py`. Within this file the following changes are required:

1. Two [YAML](http://www.yaml.org/start.html) configuration files are used
    - STJ config (for properties of jet finding metric)
    - The data configuration file is set within the STJ configuration file.
    - Examples of both can be found in the `conf/` directory.
2. Set start and end dates
3. Select sensitivity or normal run

### STJ finding Configuration: `stj_config_default.yml`


| Variable Name | Description
| ---           | ---
|`data_cfg`     | Location of data config file
|`freq`         | Input data frequency
|`zonal_opt`    | Output zonal mean (if 'mean') or individual longitude positions (if != 'mean')
|`method`       | Jet metric to use. Included are **STJPV** and **STJUMax**
|`log_file`     | Log file name and location. If `{}` is included within this string (e.g. `stj_find_{}.log`) the current time (from `datetime.now()`) at which the finder was initialised will be put into the file name (e.g. `stj_find_2017-11-02_14-08-32.log`)
|`pv_value`     | Potential vorticity level on which potential temperature is interpolated to find the jet (if using **STJPV** metric)
|`fit_deg`      | Also for **STJPV** metric, use this degree (integer) polynomial to fit the potential temperature on the `pv_value` surface
|`min_lat`      | Minimum latitude boundary (equatorward) on which to perform interpolation
|`max_lat`      | Maximum latitude boundary (poleward) on which to perform interpolation
|`update_pv`    | If isentropic PV (IPV) file(s) exist already, re-create them if this is set to `True`. If not, use files that exist
|`year_s`       | Year to start jet finding (Jan 1 of this year)
|`year_e`       | Year to end jet finding (Dec 31 of this year)
|               | Dates may also be set in `run_stj.main()` function
|`poly`         | Polynomial to use, one of 'cheby', 'legendre', or 'poly' for Chebyshev, Legendre, or polynomial fit respectively
**See comments within `conf/stj_config_default.yml` for further details**


### Data configuration: `data_config_default.yml`


| Variable Name     | Description
| ---               | ---
|`path`             | Absolute path of input data
|`wpath`            | If `path` is _not_ writeable, absolute path to directory where IPV data can be written
|`short_name`       | String name to call this dataset
|`single_var_file`  | Each variables has its own file (if True)
|`single_year_file` | Each year has its own file (if True)
|`file_paths`       | Names (within `path`) of input / output files for atmospheric variables
|                   | If `single_var_file==True` then `file_paths` has: `uwnd`, `vwnd`, `tair` (in),  and `ipv` (_output_)
|                   | If `single_var_file==False`, then `file_paths` has: `all` (in), and `ipv` (_output_)
|`lon`              | Name within netCDF file of 'longitude' variable
|`lat`              | Name within netCDF file of 'latitude' variable
|`lev`              | Name within netCDF file of 'level' variable
|`time`             | Name within netCDF file of 'time' variable
|`ztype`            | Type of levels (pressure, potential temperature, etc.)
|`pfac`             | Multiply pressure by this (float) to get units of Pascals
|`uwnd`             | Name within netCDF file of zonal wind variable
|`vwnd`             | Name within netCDF file of meridional wind variable
|`tair`             | Name within netCDF file of atmospheric temperature variable
|`ipv`              | Name within netCDF file of isentropic pv variable

**See comments within `conf/data_config_default.yml` for further details**

## How the STJPV metric works

1. The `run_stj.main()` function creates a `run_stj.JetFindRun` object, based on configuration parameters.

2. Start and end dates are set, and the `run_stj.JetFindRun.run()` method starts the run, where configuration files are checked
then the selected metric computes the jet position in each hemisphere at each time.

3. If Isentropic PV input data does not exist, this is created and written as defined in the data configuration file

4. When using the **STJPV** metric the jet is identified in the following process:

    1. Interpolate to obtain potential temperature ($\Theta$) as a function of latitude on a surface
        of constant IPV, chosen in configuration file

    2. Numerically compute meridional gradient of this surface using a polynomial fit (Chebyshev polynomials of degree 8 used by default)

    3. The jet location is determined to be at a relative maximum in the northern hemisphere, or minimum
        in the southern hemisphere of the meridional gradient of potential temperature on the PV surface at each time and longitude

    4. If multiple extrema exist, the jet latitude has the largest zonal wind shear between the
        potential vorticity surface and the lowest available level (called the "surface")

    5. The zonal mean jet position for each time is then computed as the zonal median of the
        identified positions at all longitudes, ignoring those longitudes where no position is
        identified, if the `zonal_opt` is set to `"mean"` in the configuration, otherwise the position is output at each longitude
