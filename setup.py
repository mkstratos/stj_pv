#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup

DESCRIPTION = "STJ_PV: SubTropical Jet Finding via PV Gradient Method"
LONG_DESCRIPTION = """STJ_PV Provides a framework for testing metrics of the subtopical
jet against one another."""
AUTHOR = 'Penelope Maher, Michael Kelleher'

setup(
    name='STJ_PV',
    version='1.0',
    author=AUTHOR,
    author_email='p.maher@exeter.ac.uk, kelleherme@ornl.gov',
    packages=['STJ_PV', ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "basemap>=1.1.0",
        "dask>=2.0.0",
        "matplotlib>=2.1.0",
        "netCDF4>=1.2.4",
        "numpy>=1.11.3",
        "pandas>=0.20.0",
        "psutil>=5.0.1",
        "PyYAML>=3.12",
        "scipy>=0.19.0",
        "seaborn>=0.9.0",
        "xarray>=0.9.0",
    ],
)
