# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:21:04 2018

@author: wps46139
"""

import ctypes
from sys import platform

#print(platform)

if platform == 'win32':
    mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
else:
    mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)
    
print('Using %d MKL threads' % mkl.mkl_get_max_threads())

class Cblas:
    ColMajor = 102
    NoTrans = 111
    Trans = 112
    ConjTrans = 113
