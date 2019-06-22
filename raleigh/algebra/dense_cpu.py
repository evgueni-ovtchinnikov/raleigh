# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)
# This software is distributed under a BSD licence, see ../../LICENSE.txt.

"""CPU dense algebra selector
"""

try:
    from .dense_cblas import Vectors, Matrix
    print('using mkl cblas...')
except:
    print('mkl cblas not found, using numpy...')
    from .dense_numpy import Vectors, Matrix
