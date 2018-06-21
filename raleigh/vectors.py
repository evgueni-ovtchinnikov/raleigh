# -*- coding: utf-8 -*-
"""
BLAS selector

Created on Thu Jun 14 13:08:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

try:
    from raleigh.ndarray.cblas_algebra import Vectors
    print('using mkl cblas...')
except:
    print('mkl cblas not found, using numpy...')
    from raleigh.ndarray.numpy_algebra import Vectors
