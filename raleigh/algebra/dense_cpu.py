# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""CPU dense algebra selector.
"""

from . import verbosity


try:
    from .dense_cblas import Vectors, Matrix
    if verbosity.level > 0:
        print('using mkl cblas...')
except:
    if verbosity.level > 0:
        print('mkl cblas not found, using numpy...')
    from .dense_numpy import Vectors, Matrix
