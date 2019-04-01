# -*- coding: utf-8 -*-
"""
BLAS selector

Created on Thu Jun 14 13:08:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

try:
    from .cblas_algebra import Vectors, Matrix
    print('using mkl cblas...')
except:
    print('mkl cblas not found, using numpy...')
    from .numpy_algebra import Vectors, Matrix
