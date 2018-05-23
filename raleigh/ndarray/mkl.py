# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:21:04 2018

@author: wps46139
"""

import ctypes
import numpy
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
    def __init__(self, dt):
        if dt == numpy.float32:
            self.dsize = 4
            self.gemm = mkl.cblas_sgemm
            self.axpy = mkl.cblas_saxpy
            self.copy = mkl.cblas_scopy
            self.scal = mkl.cblas_sscal
            self.norm = mkl.cblas_snrm2
            self.norm.restype = ctypes.c_float
            self.inner = mkl.cblas_sdot
            self.inner.restype = ctypes.c_float
            self.mkl_one = ctypes.c_float(1.0)
            self.mkl_zero = ctypes.c_float(0.0)
        elif dt == numpy.float64:
            self.dsize = 8
            self.gemm = mkl.cblas_dgemm
            self.axpy = mkl.cblas_daxpy
            self.copy = mkl.cblas_dcopy
            self.scal = mkl.cblas_dscal
            self.norm = mkl.cblas_dnrm2
            self.norm.restype = ctypes.c_double
            self.inner = mkl.cblas_ddot
            self.inner.restype = ctypes.c_double
            self.mkl_one = ctypes.c_double(1.0)
            self.mkl_zero = ctypes.c_double(0.0)
        elif dt == numpy.complex64:
            self.dsize = 8
            self.gemm = mkl.cblas_cgemm
            self.axpy = mkl.cblas_caxpy
            self.copy = mkl.cblas_ccopy
            self.scal = mkl.cblas_cscal
            self.norm = mkl.cblas_scnrm2
            self.norm.restype = ctypes.c_float
            self.inner = mkl.cblas_cdotc_sub
            self.cmplx_val = numpy.zeros((2,), dtype = numpy.float32)
            self.cmplx_one = numpy.zeros((2,), dtype = numpy.float32)
            self.cmplx_one[0] = 1.0
            self.cmplx_zero = numpy.zeros((2,), dtype = numpy.float32)
            self.mkl_one = ctypes.c_void_p(self.cmplx_one.ctypes.data)
            self.mkl_zero = ctypes.c_void_p(self.cmplx_zero.ctypes.data)
        elif dt == numpy.complex128:
            self.dsize = 16
            self.gemm = mkl.cblas_zgemm
            self.axpy = mkl.cblas_zaxpy
            self.copy = mkl.cblas_zcopy
            self.scal = mkl.cblas_zscal
            self.norm = mkl.cblas_dznrm2
            self.norm.restype = ctypes.c_double
            self.inner = mkl.cblas_zdotc_sub
            self.cmplx_val = numpy.zeros((2,), dtype = numpy.float64)
            self.cmplx_one = numpy.zeros((2,), dtype = numpy.float64)
            self.cmplx_one[0] = 1.0
            self.cmplx_zero = numpy.zeros((2,), dtype = numpy.float64)
            self.mkl_one = ctypes.c_void_p(self.cmplx_one.ctypes.data)
            self.mkl_zero = ctypes.c_void_p(self.cmplx_zero.ctypes.data)
        else:
            raise ValueError('data type %s not supported' % repr(dt))
