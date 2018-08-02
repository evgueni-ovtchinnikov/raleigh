# -*- coding: utf-8 -*-
"""
BLAS selector

Created on Thu Jun 14 13:08:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

raleigh_path = '..'

import sys
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

#from raleigh.ndarray.cblas_algebra import Vectors, Matrix
#print('using mkl cblas...')
try:
    from raleigh.ndarray.cblas_algebra import Vectors, Matrix
    print('using mkl cblas...')
except:
    print('mkl cblas not found, using numpy...')
    from raleigh.ndarray.numpy_algebra import Vectors, Matrix
