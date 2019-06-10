# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)
# This software is distributed under a BSD licence, see ../../LICENSE.txt.
"""CPU dense algebra selector

Created on Thu Jun 14 13:08:20 2018
"""

try:
    from .dense_cblas import Vectors, Matrix
    print('using mkl cblas...')
except:
    print('mkl cblas not found, using numpy...')
    from .dense_numpy import Vectors, Matrix
