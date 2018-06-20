# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:05:28 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes
import numpy
import sys
sys.path.append('..')

import raleigh.cuda.cuda as cuda

POINTER = ctypes.POINTER

#cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/bin'
cublas_path = cuda.cuda_path + '/cublas64_70.dll'
cublas = ctypes.CDLL(cublas_path, mode = ctypes.RTLD_GLOBAL)

create = cublas.cublasCreate_v2
create.argtypes = [POINTER(POINTER(ctypes.c_ubyte))]
create.restype = ctypes.c_int

destroy = cublas.cublasDestroy_v2
#destroy.argtypes = [POINTER(ctypes.c_ubyte)]
destroy.restype = ctypes.c_int

class Cublas:
    def __init__(self, dt):
        self.handle = POINTER(ctypes.c_ubyte)()
        err = create(ctypes.byref(self.handle))
        if err != 0:
            raise RuntimeError('cublasCreate failure')
        if dt == numpy.float32:
            self.dsize = 4
            self.norm = cublas.cublasSnrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasSdot_v2
            self.dot.restype = ctypes.c_int
        elif dt == numpy.float64:
            self.dsize = 8
            self.norm = cublas.cublasDnrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasDdot_v2
            self.dot.restype = ctypes.c_int
        else:
            raise ValueError('data type %s not supported' % repr(dt))
    def __del__(self):
        destroy(self.handle)