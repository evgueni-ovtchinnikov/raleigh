# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:21:04 2018

@author: wps46139
"""

import ctypes
import multiprocessing
import numpy
from sys import platform

#print(platform)

def _array_ptr(array, shift = 0):
    return ctypes.c_void_p(array.ctypes.data + shift)

def mkl_version(mkl):
    str_v = numpy.ndarray((128,), dtype = numpy.uint8)
    ptr_v = _array_ptr(str_v)
    len_v = ctypes.c_int(128)
    mkl.mkl_get_version_string(ptr_v, len_v)
    return str_v.tostring().decode('ascii')

if platform == 'win32':
    mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
else:
    mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)
print('Loaded %s' % mkl_version(mkl))

# find the total number of threads    
num_cpus = multiprocessing.cpu_count()

# find the number of cores using MKL threading behaviour
mkl.MKL_Set_Dynamic(1)
# now mkl_get_max_threads() returns the number of cores!
num_cores = mkl.mkl_get_max_threads()

# if hyperthreading is used, increase the number of mkl threads
# to achieve slightly better performance
if num_cpus == 2*num_cores:
    num_threads = num_cpus - 1
    #mkl.mkl_set_dynamic(ctypes.byref(ctypes.c_int(0)))
    #mkl.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))
    mkl.MKL_Set_Dynamic(0)
    mkl.MKL_Set_Num_Threads(num_threads)

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

class SparseSymmetricMatrix:
    def __init__(self, a, ia, ja):
        self.__a = self.__array_ptr(a)
        self.__ib = self.__array_ptr(ia)
        self.__ie = self.__array_ptr(ia[1:])
        self.__ja = self.__array_ptr(ja)
        self.__n = ctypes.c_int(ia.shape[0] - 1)
        self.__dtype = a.dtype
        self.__oz = numpy.array([1.0, 0.0]).astype(a.dtype)
        self.__one = self.__array_ptr(self.__oz)
        self.__zero = self.__array_ptr(self.__oz[1:])
        if a.dtype == numpy.float32:
            self.__csrmm = mkl.mkl_scsrmm
            self.__csrsymv = mkl.mkl_scsrsymv
        elif a.dtype == numpy.float64:
            self.__csrmm = mkl.mkl_dcsrmm
            self.__csrsymv = mkl.mkl_dcsrsymv
        elif a.dtype == numpy.complex64:
            self.__csrmm = mkl.mkl_ccsrmm
            self.__csrsymv = mkl.mkl_ccsrsymv
        elif a.dtype == numpy.complex128:
            self.__csrmm = mkl.mkl_zcsrmm
            self.__csrsymv = mkl.mkl_zcsrsymv
        else:
            raise ValueError('unsupported data type')
        self.__csrmm.restype = None
        uplo = 'U'
        self.__u = ctypes.c_char_p(uplo.encode('utf-8'))
        trans = 'N'
        self.__t = ctypes.c_char_p(trans.encode('utf-8'))
        descr = 'SUNF  '
        self.__d = ctypes.c_char_p(descr.encode('utf-8'))
    def __array_ptr(self, array, shift = 0):
        return ctypes.c_void_p(array.ctypes.data + shift)
    def dot(self, x, y):
        ptr_x = self.__array_ptr(x)
        ptr_y = self.__array_ptr(y)
        s = x.shape
        if len(s) == 1 or s[0] == 1:
            #print('using *csrsymv...')
            if len(s) == 1:
                n = s[0]
            else:
                n = numpy.prod(s[1:])
            n = ctypes.c_int(n)
            self.__csrsymv(self.__u, ctypes.byref(n), \
                           self.__a, self.__ib, self.__ja, ptr_x, ptr_y)
            return
        #print('using *csrmm...')
        m, n = s[0], numpy.prod(s[1:])
        mn = numpy.array([m, n])
        ptr_m = self.__array_ptr(mn)
        ptr_n = self.__array_ptr(mn[1:])
        n = self.__n
        self.__csrmm(self.__t, ctypes.byref(n), ptr_m, ctypes.byref(n), \
            self.__one, self.__d, self.__a, self.__ja, self.__ib, self.__ie, \
            ptr_x, ptr_n, self.__zero, ptr_y, ptr_n)

class SparseSymmetricDirectSolver:
    def __init__(self, dtype = numpy.float64):
        self.__MKL_DSS_SUCCESS = 0
        self.__MKL_DSS_SYMMETRIC         = 536870976 # 0x20000040
        self.__MKL_DSS_SYMMETRIC_COMPLEX = 536871168 # 0x20000100
        # NO OPTION FOR HERMITIAN MATRIX !!!!!!!!!!!!!!!!!!!!!!!!
        self.__MKL_DSS_POSITIVE_DEFINITE = 134217792 # 0x08000040
        self.__MKL_DSS_INDEFINITE        = 134217856 # 0x08000080
        self.__MKL_DSS_GET_ORDER         = 268435712 # 0x10000100
        self.__MKL_DSS_MY_ORDER          = 268435584 # 0x10000080
        self.__MKL_DSS_SINGLE_PRECISION  =     65536 # 0x00010000
        self.__handle = ctypes.POINTER(ctypes.c_ubyte)()
        self.__dtype = dtype
        self.__n = None
        self.__ia = None
        self.__real = None
        self.__order = None
        self.__scale = None
        self.__dss_define_structure = mkl.dss_define_structure
        self.__dss_define_structure.restype = ctypes.c_int
        self.__dss_reorder = mkl.dss_reorder
        self.__dss_reorder.restype = ctypes.c_int
        self.__dss_factor_real = mkl.dss_factor_real
        self.__dss_factor_real.restype = ctypes.c_int
        self.__dss_factor_complex = mkl.dss_factor_complex
        self.__dss_factor_complex.restype = ctypes.c_int
        self.__dss_solve_real = mkl.dss_solve_real
        self.__dss_solve_real.restype = ctypes.c_int
        self.__dss_solve_complex = mkl.dss_solve_complex
        self.__dss_solve_complex.restype = ctypes.c_int
        self.__dss_statistics = mkl.dss_statistics
        self.__dss_statistics.restype = ctypes.c_int
        self.__dss_delete = mkl.dss_delete
        self.__dss_delete.restype = ctypes.c_int
        if dtype == numpy.float64 or dtype == numpy.complex128:
            opt = ctypes.c_int(0)
        else:
            opt = ctypes.c_int(self.__MKL_DSS_SINGLE_PRECISION)
        mkl.dss_create(ctypes.byref(self.__handle), ctypes.byref(opt))
    def __del__(self):
        #import ctypes
        opt = ctypes.c_int(0)
        err = self.__dss_delete(ctypes.byref(self.__handle), ctypes.byref(opt))
    def __array_ptr(self, array, shift = 0):
        return ctypes.c_void_p(array.ctypes.data + shift)
    def handle(self):
        return self.__handle
    def define_structure(self, ia, ja):
        rows = ia.shape[0] - 1
        nonzeros = ia[rows] - ia[0]
        n = ctypes.c_int(rows)
        nnz = ctypes.c_int(nonzeros)
        ptr_ia = self.__array_ptr(ia)
        ptr_ja = self.__array_ptr(ja)
        if self.__dtype == numpy.float32 or self.__dtype == numpy.float64:
            sym = ctypes.c_int(self.__MKL_DSS_SYMMETRIC)
        else:
            sym = ctypes.c_int(self.__MKL_DSS_SYMMETRIC_COMPLEX)
        err = self.__dss_define_structure(ctypes.byref(self.__handle), \
            ctypes.byref(sym), \
            ptr_ia, ctypes.byref(n), ctypes.byref(n), \
            ptr_ja, ctypes.byref(nnz))
        if err is not self.__MKL_DSS_SUCCESS:
            raise RuntimeError('mkl.dss_define_structure failed')
        self.__n = rows
        self.__ia = ia
        self.__ja = ja
        self.__order = numpy.ndarray((rows,), dtype = numpy.int32)
    def reorder(self, order = None):
        if self.__n is None:
            raise RuntimeError('call to mkl.dss_define_structure missing')
        if order is None:
            ptr_order = self.__array_ptr(self.__order)
            get_order = ctypes.c_int(self.__MKL_DSS_GET_ORDER)
            err = self.__dss_reorder(ctypes.byref(self.__handle), \
                                     ctypes.byref(get_order), \
                                     ptr_order)
        else:
            if order.size != n:
                raise RuntimeError('mkl.dss_reorder given wrong order')
            self.__order = order.copy()
            ptr_order = self.__array_ptr(order)
            my_order = ctypes.c_int(self.__MKL_DSS_MY_ORDER)
            err = self.__dss_reorder(ctypes.byref(self.__handle), \
                                     ctypes.byref(my_order), \
                                     ptr_order)
        if err is not self.__MKL_DSS_SUCCESS:
            raise RuntimeError('mkl.dss_reorder failed')
    def factorize(self, a, sigma = 0, pos_def = False): #, scale = True):
        if sigma == 0.0: # and not scale:
            a_s = a
##            self.__scale = None
        else:
            a_s = a.copy()
            if sigma != 0.0:
                a_s[self.__ia[:-1] - 1] -= sigma
##            if scale:
##                d = a_s[self.__ia[:-1] - 1]
##                s = numpy.sqrt(abs(d))
##                rows = self.__ia.shape[0] - 1
##                for row in range(rows):
##                    for i in range(self.__ia[row], self.__ia[row + 1]):
##                        col = self.__ja[i - 1] - 1
##                        sc = s[row]*s[col]
##                        if sc != 0.0: a_s[i - 1] /= sc
##                self.__scale = s
##            else:
##                self.__scale = None
        if pos_def:
            pd = ctypes.c_int(self.__MKL_DSS_POSITIVE_DEFINITE)
        else:
            pd = ctypes.c_int(self.__MKL_DSS_INDEFINITE)
        if a.dtype.kind == 'c':
            ptr_as = self.__array_ptr(a_s)
            err = self.__dss_factor_complex(ctypes.byref(self.__handle), \
                                            ctypes.byref(pd), ptr_as)
            if err is not self.__MKL_DSS_SUCCESS:
                raise RuntimeError('mkl.dss_factor_complex failed')
            self.__real = False
        else:
            ptr_as = self.__array_ptr(a_s)
            err = self.__dss_factor_real(ctypes.byref(self.__handle), \
                                         ctypes.byref(pd), ptr_as)
            if err is not self.__MKL_DSS_SUCCESS:
                raise RuntimeError('mkl.dss_factor_real failed')
            self.__real = True
    def solve(self, b, x):
        if self.__real is None:
            raise RuntimeError('solve: factorization not performed')
        opt = ctypes.c_int(0)
        if len(b.shape) > 1:
            nrhs = b.shape[0]
        else:
            nrhs = 1
        m = ctypes.c_int(nrhs)
        if self.__scale is None:
            ptr_b = self.__array_ptr(b)
        else:
            b_s = b/self.__scale
            ptr_b = self.__array_ptr(b_s)
        ptr_x = self.__array_ptr(x)
        if self.__real:
            err = self.__dss_solve_real(ctypes.byref(self.__handle), \
                                        ctypes.byref(opt), \
                                        ptr_b, ctypes.byref(m), ptr_x)
            if err is not self.__MKL_DSS_SUCCESS:
                raise RuntimeError('mkl.dss_solve_real failed')
        else:
            err = self.__dss_solve_complex(ctypes.byref(self.__handle), \
                                           ctypes.byref(opt), \
                                           ptr_b, ctypes.byref(m), ptr_x)
            if err is not self.__MKL_DSS_SUCCESS:
                raise RuntimeError('mkl.dss_solve_complex failed')
        if self.__scale is not None:
            x /= self.__scale
    def inertia(self):
        # does not work for a Hermitian matrix
        opt = ctypes.c_int(0)
        inertia = numpy.ndarray((3,))
        stat = "Inertia"
        ptr_stat = ctypes.c_char_p(stat.encode('utf-8'))
        ptr_inertia = self.__array_ptr(inertia)
        err = self.__dss_statistics(ctypes.byref(self.__handle), \
                             ctypes.byref(opt), \
                             ptr_stat, ptr_inertia)
        if err is not self.__MKL_DSS_SUCCESS:
            raise RuntimeError('mkl.dss_statistics failed')
        return inertia

