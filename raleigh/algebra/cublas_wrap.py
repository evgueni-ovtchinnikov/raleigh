# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""CUBLAS loader/wrapper.
"""

import ctypes
import glob
import numpy
from sys import platform

from . import verbosity


POINTER = ctypes.POINTER


try:
    if platform == 'win32':
        from .cuda_wrap import cuda_path
        cublas_dll = glob.glob(cuda_path + '/cublas64*')[0]
        cublas = ctypes.CDLL(cublas_dll, mode=ctypes.RTLD_GLOBAL)
        if verbosity.level > 0:
            print('loaded %s' % cublas_dll)
    else:
        cublas = ctypes.CDLL('libcublas.so', mode=ctypes.RTLD_GLOBAL)

    cublasCreate = cublas.cublasCreate_v2
    cublasCreate.argtypes = [POINTER(POINTER(ctypes.c_ubyte))]
    cublasCreate.restype = ctypes.c_int

    cublasDestroy = cublas.cublasDestroy_v2
    cublasDestroy.restype = ctypes.c_int

    cublasVersion = cublas.cublasGetVersion_v2
    cublasVersion.restype = ctypes.c_int

    cublas_handle = POINTER(ctypes.c_ubyte)()
    err = cublasCreate(ctypes.byref(cublas_handle))
    v = ctypes.c_int()
    err = cublasVersion(cublas_handle, ctypes.byref(v))
    version = v.value
    if verbosity.level > 0:
        print('CUBLAS version: %d' % version)
    cublasDestroy(cublas_handle)

except:
    if verbosity.level > 0:
        print('CUBLAS not found, switching to cpu...')
    raise RuntimeError('CUBLAS not found')

##try:
##    if platform == 'win32':
##        from .cuda_wrap import cuda_path
##        cusolver_dll = glob.glob(cuda_path + '/cusolver64*')[0]
##        cusolver = ctypes.CDLL(cusolver_dll, mode=ctypes.RTLD_GLOBAL)
##        if verbosity.level > 0:
##            print('loaded %s' % cusolver_dll)
##    else:
##        cusolver = ctypes.CDLL('libcusolver.so', mode=ctypes.RTLD_GLOBAL)
##
##    cusolverCreate = cusolver.cusolverDnCreate
##    cusolverCreate.argtypes = [POINTER(POINTER(ctypes.c_ubyte))]
##    cusolverCreate.restype = ctypes.c_int
##
##    cusolverDestroy = cusolver.cusolverDnDestroy
##    cusolverDestroy.restype = ctypes.c_int
##
##except:
##    if verbosity.level > -1:
##        print('cuSOLVER library not found')
##    raise RuntimeError('CUSOLVER not found')


class Cublas:
    '''CUBLAS wrapper.
    '''

    CtypesPtr = POINTER(ctypes.c_ubyte)
    NoTrans = 0
    Trans = 1
    ConjTrans = 2
    HostMode = ctypes.c_int(0)
    DeviceMode = ctypes.c_int(1)

    def __init__(self, dt):
#        print('creating cublas...')
        self.handle = Cublas.CtypesPtr()
        self.cusolver_handle = Cublas.CtypesPtr()
        err = cublasCreate(ctypes.byref(self.handle))
        if err != 0:
            raise RuntimeError('cublasCreate failure, cuda error %d' % err)
##        err = cusolverCreate(ctypes.byref(self.cusolver_handle))
##        if err != 0:
##            raise RuntimeError('cusolverCreate failure, cuda error %d' % err)
        self.getPointerMode = cublas.cublasGetPointerMode_v2
        self.getPointerMode.restype = ctypes.c_int
        self.setPointerMode = cublas.cublasSetPointerMode_v2
        self.setPointerMode.restype = ctypes.c_int
        self.setStream = cublas.cublasSetStream_v2
        self.setStream.restype = ctypes.c_int
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
            self.gemm_batched = cublas.cublasSgemmBatched
            self.gemm_batched.restype = ctypes.c_int
##            self.svd_buff_size = cusolver.cusolverDnSgesvd_bufferSize
##            self.svd = cusolver.cusolverDnSgesvd
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
            self.gemm_batched = cublas.cublasDgemmBatched
            self.gemm_batched.restype = ctypes.c_int
##            self.svd_buff_size = cusolver.cusolverDnDgesvd_bufferSize
##            self.svd = cusolver.cusolverDnDgesvd
        elif dt == numpy.complex64:
            self.dsize = 8
            self.copy = cublas.cublasCcopy_v2
            self.copy.restype = ctypes.c_int
            self.axpy = cublas.cublasCaxpy_v2
            self.axpy.restype = ctypes.c_int
            self.scal = cublas.cublasCscal_v2
            self.scal.restype = ctypes.c_int
            self.fscal = cublas.cublasSscal_v2
            self.fscal.restype = ctypes.c_int
            self.norm = cublas.cublasScnrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasCdotc_v2
            self.dot.restype = ctypes.c_int
            self.gemm = cublas.cublasCgemm_v2
            self.gemm.restype = ctypes.c_int
            self.gemm_batched = cublas.cublasCgemmBatched
            self.gemm_batched.restype = ctypes.c_int
##            self.svd_buff_size = cusolver.cusolverDnCgesvd_bufferSize
##            self.svd = cusolver.cusolverDnCgesvd
        elif dt == numpy.complex128:
            self.dsize = 16
            self.copy = cublas.cublasZcopy_v2
            self.copy.restype = ctypes.c_int
            self.axpy = cublas.cublasZaxpy_v2
            self.axpy.restype = ctypes.c_int
            self.scal = cublas.cublasZscal_v2
            self.scal.restype = ctypes.c_int
            self.fscal = cublas.cublasDscal_v2
            self.fscal.restype = ctypes.c_int
            self.norm = cublas.cublasDznrm2_v2
            self.norm.restype = ctypes.c_int
            self.dot = cublas.cublasZdotc_v2
            self.dot.restype = ctypes.c_int
            self.gemm = cublas.cublasZgemm_v2
            self.gemm.restype = ctypes.c_int
            self.gemm_batched = cublas.cublasZgemmBatched
            self.gemm_batched.restype = ctypes.c_int
##            self.svd_buff_size = cusolver.cusolverDnZgesvd_bufferSize
##            self.svd = cusolver.cusolverDnZgesvd
        else:
            raise ValueError('data type %s not supported' % repr(dt))
##        self.svd_buff_size.restype = ctypes.c_int
##        self.svd.argtypes = [POINTER(ctypes.c_ubyte), \
##            ctypes.c_char, ctypes.c_char, ctypes.c_int, ctypes.c_int, \
##            POINTER(ctypes.c_ubyte), ctypes.c_int, POINTER(ctypes.c_ubyte), \
##            POINTER(ctypes.c_ubyte), ctypes.c_int, \
##            POINTER(ctypes.c_ubyte), ctypes.c_int, \
##            POINTER(ctypes.c_ubyte), ctypes.c_int, \
##            POINTER(ctypes.c_ubyte), POINTER(ctypes.c_ubyte)]
##        self.svd.restype = ctypes.c_int
        self.all = ctypes.c_char('A'.encode('utf-8'))
        self.small = ctypes.c_char('S'.encode('utf-8'))
        self.ovwrt = ctypes.c_char('O'.encode('utf-8'))

    def __del__(self):
#        print('destroying cublas...')
        cublasDestroy(self.handle)
##        cusolverDestroy(self.cusolver_handle)
