# -*- coding: utf-8 -*-
"""
Dense algebra on cpu selector

Created on Thu Jun 14 13:08:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

try:
    from .dense_cblas import Vectors, Matrix
    print('using mkl cblas...')
except:
    print('mkl cblas not found, using numpy...')
    from .dense_numpy import Vectors, Matrix
