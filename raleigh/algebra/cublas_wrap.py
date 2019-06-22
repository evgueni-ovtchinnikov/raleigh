# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)
# This software is distributed under a BSD licence, see ../../LICENSE.txt.

"""CUBLAS wrapper.
"""

import ctypes
import glob
import numpy
from sys import platform


POINTER = ctypes.POINTER


try:
    if platform == 'win32':
        from .cuda_wrap import cuda_path
        cublas_dll = glob.glob(cuda_path + '/cublas64*')[0]
        cublas = ctypes.CDLL(cublas_dll, mode=ctypes.RTLD_GLOBAL)
    else:
        cublas = ctypes.CDLL('libcublas.so', mode=ctypes.RTLD_GLOBAL)

    create = cublas.cublasCreate_v2
    create.argtypes = [POINTER(POINTER(ctypes.c_ubyte))]
    create.restype = ctypes.c_int

    destroy = cublas.cublasDestroy_v2
    destroy.restype = ctypes.c_int

    cublasVersion = cublas.cublasGetVersion_v2
    cublasVersion.restype = ctypes.c_int

    cublas_handle = POINTER(ctypes.c_ubyte)()
    err = create(ctypes.byref(cublas_handle))
    v = ctypes.c_int()
    err = cublasVersion(cublas_handle, ctypes.byref(v))
    version = v.value
    print('CUBLAS version: %d' % version)
    destroy(cublas_handle)
except:
    print('CUBLAS not found, switching to cpu...')
    raise RuntimeError('CUBLAS not found')


class Cublas:
    NoTrans = 0
    Trans = 1
    ConjTrans = 2
    def __init__(self, dt):
        self.handle = POINTER(ctypes.c_ubyte)()
        err = create(ctypes.byref(self.handle))
        if err != 0:
            raise RuntimeError('cublasCreate failure')
        if dt == numpy.float32:
            self.dsize = 4
            self.copy = cublas.cublasScopy_v2
            self.copy.restype = ctypes.c_int
            self.axpy = cublas.cublasSaxpy_v2
            self.axpy.restype = ctypes.c_int
            self.scal = cublas.cublasSscal_v2
            self.scal.restype = ctypes.c_int
            self.norm = cublas.cublasSnrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasSdot_v2
            self.dot.restype = ctypes.c_int
            self.gemm = cublas.cublasSgemm_v2
            self.gemm.restype = ctypes.c_int
        elif dt == numpy.float64:
            self.dsize = 8
            self.copy = cublas.cublasDcopy_v2
            self.copy.restype = ctypes.c_int
            self.axpy = cublas.cublasDaxpy_v2
            self.axpy.restype = ctypes.c_int
            self.scal = cublas.cublasDscal_v2
            self.scal.restype = ctypes.c_int
            self.norm = cublas.cublasDnrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasDdot_v2
            self.dot.restype = ctypes.c_int
            self.gemm = cublas.cublasDgemm_v2
            self.gemm.restype = ctypes.c_int
        elif dt == numpy.complex64:
            self.dsize = 8
            self.copy = cublas.cublasCcopy_v2
            self.copy.restype = ctypes.c_int
            self.axpy = cublas.cublasCaxpy_v2
            self.axpy.restype = ctypes.c_int
            self.scal = cublas.cublasCscal_v2
            self.scal.restype = ctypes.c_int
            self.norm = cublas.cublasScnrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasCdotc_v2
            self.dot.restype = ctypes.c_int
            self.gemm = cublas.cublasCgemm_v2
            self.gemm.restype = ctypes.c_int
        elif dt == numpy.complex128:
            self.dsize = 16
            self.copy = cublas.cublasZcopy_v2
            self.copy.restype = ctypes.c_int
            self.axpy = cublas.cublasZaxpy_v2
            self.axpy.restype = ctypes.c_int
            self.scal = cublas.cublasZscal_v2
            self.scal.restype = ctypes.c_int
            self.norm = cublas.cublasDznrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasZdotc_v2
            self.dot.restype = ctypes.c_int
            self.gemm = cublas.cublasZgemm_v2
            self.gemm.restype = ctypes.c_int
        else:
            raise ValueError('data type %s not supported' % repr(dt))
    def __del__(self):
        destroy(self.handle)
