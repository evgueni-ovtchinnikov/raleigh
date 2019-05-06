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
        self.__a = a
        self.__ib = ia
        self.__ie = ia[1:]
        self.__ja = ja
        self.__n = ctypes.c_int(ia.shape[0] - 1)
        self.__dtype = a.dtype
        self.__oz = numpy.array([1.0, 0.0]).astype(a.dtype)
        self.__one = _array_ptr(self.__oz)
        self.__zero = _array_ptr(self.__oz[1:])
        self.__complex = False
        if a.dtype == numpy.float32:
            self.__csrmm = mkl.mkl_scsrmm
            self.__csrsymv = mkl.mkl_scsrsymv
            self.__descr = 'SUNF  '
        elif a.dtype == numpy.float64:
            self.__csrmm = mkl.mkl_dcsrmm
            self.__csrsymv = mkl.mkl_dcsrsymv
            self.__descr = 'SUNF  '
        elif a.dtype == numpy.complex64:
            self.__csrmm = mkl.mkl_ccsrmm
            self.__csrsymv = mkl.mkl_ccsrsymv
            self.__descr = 'HUNF  '
            self.__complex = True
        elif a.dtype == numpy.complex128:
            self.__csrmm = mkl.mkl_zcsrmm
            self.__csrsymv = mkl.mkl_zcsrsymv
            self.__descr = 'HUNF  '
            self.__complex = True
        else:
            raise ValueError('unsupported data type')
        self.__csrmm.restype = None
        self.__uplo = 'U'
        self.__trans = 'N'
        self.__u = ctypes.c_char_p(self.__uplo.encode('utf-8'))
        self.__t = ctypes.c_char_p(self.__trans.encode('utf-8'))
        self.__d = ctypes.c_char_p(self.__descr.encode('utf-8'))
    def dot(self, x, y):
        ptr_x = _array_ptr(x)
        ptr_y = _array_ptr(y)
        ptr_a = _array_ptr(self.__a)
        ptr_ib = _array_ptr(self.__ib)
        ptr_ie = _array_ptr(self.__ie)
        ptr_ja = _array_ptr(self.__ja)
        s = x.shape
        if (len(s) == 1 or s[0] == 1) and not self.__complex:
            #print('using *csrsymv...')
            if len(s) == 1:
                n = s[0]
            else:
                n = numpy.prod(s[1:])
            n = ctypes.c_int(n)
            self.__csrsymv(self.__u, ctypes.byref(n), \
                           ptr_a, ptr_ib, ptr_ja, ptr_x, ptr_y)
            return
        #print('using *csrmm...')
        if len(s) > 1:
            m, n = s[0], int(numpy.prod(s[1:]))
        else:
            m = 1
            n = self.__n.value
        mn = numpy.array([m, n])
        ptr_m = _array_ptr(mn)
        ptr_n = _array_ptr(mn[1:])
        n = self.__n
        self.__csrmm(self.__t, ctypes.byref(n), ptr_m, ctypes.byref(n), \
            self.__one, self.__d, ptr_a, ptr_ja, ptr_ib, ptr_ie, \
            ptr_x, ptr_n, self.__zero, ptr_y, ptr_n)

class ILUT:
    def __init__(self, a, ia, ja):
        self.__a = a
        self.__ia = ia
        self.__ja = ja
        n = ia.shape[0] - 1
        nnz = ia[n] - ia[0]
        self.__rows = n
        self.__nnz = nnz//n
        self.__ipar = numpy.zeros((128,), dtype = numpy.int32)
        self.__dpar = numpy.zeros((128,))
        self.__w = numpy.zeros((n,))
        self.__ipar[0] = n
        self.__b = None
        self.__ib = None
        self.__jb = None
        self.__u = 'U'
        self.__l = 'L'
        self.__n = 'N'
        self.__U = ctypes.c_char_p(self.__u.encode('utf-8'))
        self.__L = ctypes.c_char_p(self.__l.encode('utf-8'))
        self.__N = ctypes.c_char_p(self.__n.encode('utf-8'))
    def factorize(self, tol = 1e-2, max_fill_rel = 10):
        max_fill_abs = min(self.__rows - 1, self.__nnz * max_fill_rel)
        self.__dpar[30] = tol
        n = self.__rows
        nnz = (2*max_fill_abs + 1)*n
        print(n, nnz)
        self.__b = numpy.ndarray((nnz,)) 
        self.__ib = numpy.ndarray((n + 1,)) 
        self.__jb = numpy.ndarray((nnz,), dtype = numpy.int32)
        n_c = ctypes.c_int(n)
        mf_c = ctypes.c_int(max_fill_abs)
        ierr_c = ctypes.c_int()
        tol_c = ctypes.c_double(tol)
        ptr_a = _array_ptr(self.__a)
        ptr_ia = _array_ptr(self.__ia)
        ptr_ja = _array_ptr(self.__ja)
        ptr_b = _array_ptr(self.__b)
        ptr_ib = _array_ptr(self.__ib)
        ptr_jb = _array_ptr(self.__jb)
        ptr_ipar = _array_ptr(self.__ipar)
        ptr_dpar = _array_ptr(self.__dpar)
        mkl.dcsrilut(ctypes.byref(n_c), \
                     ptr_a, ptr_ia, ptr_ja, ptr_b, ptr_ib, ptr_jb, \
                     ctypes.byref(tol_c), ctypes.byref(mf_c), \
                     ptr_ipar, ptr_dpar, ctypes.byref(ierr_c))
        if ierr_c.value != 0: print(ierr_c.value)
    def solve(self, f, x):
        m, n = f.shape
        n = ctypes.c_int(self.__rows)
        print(n)
        ptr_b = _array_ptr(self.__b)
        ptr_ib = _array_ptr(self.__ib)
        ptr_jb = _array_ptr(self.__jb)
        ptr_w = _array_ptr(self.__w)
        for i in range(m):
            ptr_f = _array_ptr(f[i,:])
            ptr_x = _array_ptr(x[i,:])
            mkl.mkl_dcsrtrsv(self.__L, self.__N, self.__U, ctypes.byref(n), \
                             ptr_b, ptr_ib, ptr_jb, ptr_f, ptr_w)
            mkl.mkl_dcsrtrsv(self.__U, self.__N, self.__N, ctypes.byref(n), \
                             ptr_b, ptr_ib, ptr_jb, ptr_w, ptr_x)

class ParDiSo:
    def __init__(self, dtype = numpy.float64, pos_def = False):
        self.__pardiso = mkl.pardiso
        self.__pt = numpy.ndarray((64,), dtype = numpy.int64)
        self.__iparm = numpy.ndarray((64,), dtype = numpy.int32)
        self.__handle = _array_ptr(self.__pt)
        if (dtype == numpy.float32 or dtype == numpy.float64):
            m = 2
            self.__real = True
        elif (dtype == numpy.complex64 or dtype == numpy.complex128):
            m = 4
            self.__real = False
        else:
            raise ValueError('ParDiSo constructor: wrong data type')
        if not pos_def:
            m = -m
        #print('matrix type: %d' % m)
        mtype = ctypes.c_int(m)
        self.__dtype = dtype
        self.__mtype = mtype
        self.__a = None
        self.__ia = None
        self.__ja = None
        self.__perm = None
        self.__dummy = numpy.ndarray((1,))
        self.__ptr = ctypes.c_void_p(self.__dummy.ctypes.data)
        self.__ptr_iparm = _array_ptr(self.__iparm)
        mkl.pardisoinit(self.__handle, ctypes.byref(mtype), self.__ptr_iparm)
        self.__iparm[4] = 2
        if dtype == numpy.float32 or dtype == numpy.complex64:
            #print('single precision')
            self.__iparm[27] = 1
    def __del__(self):
        step = ctypes.c_int(-1)
        maxf = ctypes.c_int(1)
        mnum = ctypes.c_int(1)
        n = ctypes.c_int(1)
        m = ctypes.c_int(1)
        ptr = self.__ptr
        verb = ctypes.c_int(0)
        err = ctypes.c_long(0)
        self.__pardiso(self.__handle, \
            ctypes.byref(maxf), ctypes.byref(mnum), \
            ctypes.byref(self.__mtype), \
            ctypes.byref(step), ctypes.byref(n), ptr, ptr, ptr, \
            ptr, ctypes.byref(m), self.__ptr_iparm, ctypes.byref(verb), \
            ptr, ptr, ctypes.byref(err))
        if err.value != 0: print(err.value)
    def handle(self):
        return self.__handle
    def iparm(self):
        return self.__iparm
    def perm(self):
        return self.__perm
    def analyse(self, a, ia, ja):
        #print('iparm[27]: %d' % self.__iparm[27])
        self.__a = a
        self.__ia = ia
        self.__ja = ja
        step = ctypes.c_int(11)
        maxf = ctypes.c_int(1)
        mnum = ctypes.c_int(1)
        rows = ia.shape[0] - 1
        n = ctypes.c_int(rows)
        m = ctypes.c_int(1)
        ptr_a = _array_ptr(self.__a)
        ptr_ia = _array_ptr(self.__ia)
        ptr_ja = _array_ptr(self.__ja)
        self.__perm = numpy.ndarray((rows,), dtype = numpy.int32)
        ptr_perm = _array_ptr(self.__perm)
        ptr_iparm = _array_ptr(self.__iparm)
        verb = ctypes.c_int(0)
        err = ctypes.c_long(0)
        self.__pardiso(self.__handle, ctypes.byref(maxf), ctypes.byref(mnum), \
                    ctypes.byref(self.__mtype), \
                    ctypes.byref(step), ctypes.byref(n), ptr_a, ptr_ia, ptr_ja, \
                    ptr_perm, ctypes.byref(m), ptr_iparm, ctypes.byref(verb), \
                    self.__ptr, self.__ptr, ctypes.byref(err))
        if err.value != 0: print(err.value)
        #print(self.__perm)
    def factorize(self):
        #print('iparm[27]: %d' % self.__iparm[27])
        step = ctypes.c_int(22)
        maxf = ctypes.c_int(1)
        mnum = ctypes.c_int(1)
        rows = self.__ia.shape[0] - 1
        n = ctypes.c_int(rows)
        m = ctypes.c_int(1)
        ptr_a = _array_ptr(self.__a)
        ptr_ia = _array_ptr(self.__ia)
        ptr_ja = _array_ptr(self.__ja)
        ptr_perm = _array_ptr(self.__perm)
        ptr_iparm = _array_ptr(self.__iparm)
        verb = ctypes.c_int(0)
        err = ctypes.c_long(0)
        self.__pardiso(self.__handle, ctypes.byref(maxf), ctypes.byref(mnum), \
                    ctypes.byref(self.__mtype), \
                    ctypes.byref(step), ctypes.byref(n), ptr_a, ptr_ia, ptr_ja, \
                    ptr_perm, ctypes.byref(m), ptr_iparm, ctypes.byref(verb), \
                    self.__ptr, self.__ptr, ctypes.byref(err))
        if err.value != 0: print(err.value)
    def solve(self, b, x, part = None):
        #print('iparm[27]: %d' % self.__iparm[27])
        if len(b.shape) > 1:
            nrhs = b.shape[0]
        else:
            nrhs = 1
        if part == 'f':
            step = ctypes.c_int(331)
        elif part == 'd':
            step = ctypes.c_int(332)
        elif part == 'b':
            step = ctypes.c_int(333)
        else:
            step = ctypes.c_int(33)
        maxf = ctypes.c_int(1)
        mnum = ctypes.c_int(1)
        rows = self.__ia.shape[0] - 1
        n = ctypes.c_int(rows)
        m = ctypes.c_int(nrhs)
        ptr_a = _array_ptr(self.__a)
        ptr_ia = _array_ptr(self.__ia)
        ptr_ja = _array_ptr(self.__ja)
        ptr_perm = _array_ptr(self.__perm)
        ptr_iparm = _array_ptr(self.__iparm)
        verb = ctypes.c_int(0)
        err = ctypes.c_long(0)
        ptr_b = _array_ptr(b)
        ptr_x = _array_ptr(x)
        self.__pardiso(self.__handle, ctypes.byref(maxf), ctypes.byref(mnum), \
                    ctypes.byref(self.__mtype), \
                    ctypes.byref(step), ctypes.byref(n), ptr_a, ptr_ia, ptr_ja, \
                    ptr_perm, ctypes.byref(m), ptr_iparm, ctypes.byref(verb), \
                    ptr_b, ptr_x, ctypes.byref(err))
        if err.value != 0: print(err.value)
    def diag(self):
        rows = self.__ia.shape[0] - 1
        perm = self.__perm - 1
        f = numpy.zeros((4, rows), dtype = self.__dtype)
        w = numpy.zeros((4, rows), dtype = self.__dtype)
        for i in range(rows):
            j = i%4
            f[j, perm[i]] = 1
        self.solve(f, w, part = 'd')
        w = w[:,perm]
        d = numpy.zeros((2, rows), dtype = self.__dtype)
        i = 0
        while i < rows:
            if i > 0:
                d[1, i - 1] = w[0, i - 1]
            d[0, i] = w[0, i]
            d[1, i] = w[1, i]
            if i == rows - 1:
                break
            i += 1
            d[0, i] = w[1, i]
            d[1, i] = w[2, i]
            if i == rows - 1:
                break
            i += 1
            d[0, i] = w[2, i]
            d[1, i] = w[3, i]
            if i == rows - 1:
                break
            i += 1
            d[0, i] = w[3, i]
            i += 1
        return d
    def inertia(self):
        if self.__real:
            return self.__iparm[22], self.__iparm[21]
        rows = self.__ia.shape[0] - 1
        diags = self.diag()
        diag = numpy.real(diags[0, :])
        offd = diags[1, :]
        nneg = 0
        npos = 0
        i = 0
        while i < rows:
            a = diag[i]
            c = offd[i]
            if c == 0 or i == rows - 1:
                if a < 0:
                    nneg += 1
                elif a > 0:
                    npos += 1
                i += 1
            else:
                b = diag[i + 1]
                det = a*b - abs(c)**2
                s = a + b
                if det == 0:
                    if s < 0:
                        nneg += 1
                    elif s > 0:
                        npos += 1
                elif det < 0:
                    nneg += 1
                    npos += 1
                elif s < 0:
                    nneg += 2
                i += 2
        return nneg, npos

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
    def handle(self):
        return self.__handle
    def define_structure(self, ia, ja):
        rows = ia.shape[0] - 1
        nonzeros = ia[rows] - ia[0]
        n = ctypes.c_int(rows)
        nnz = ctypes.c_int(nonzeros)
        ptr_ia = _array_ptr(ia)
        ptr_ja = _array_ptr(ja)
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
            ptr_order = _array_ptr(self.__order)
            get_order = ctypes.c_int(self.__MKL_DSS_GET_ORDER)
            err = self.__dss_reorder(ctypes.byref(self.__handle), \
                                     ctypes.byref(get_order), \
                                     ptr_order)
        else:
            if order.size != n:
                raise RuntimeError('mkl.dss_reorder given wrong order')
            self.__order = order.copy()
            ptr_order = _array_ptr(order)
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
            ptr_as = _array_ptr(a_s)
            err = self.__dss_factor_complex(ctypes.byref(self.__handle), \
                                            ctypes.byref(pd), ptr_as)
            if err is not self.__MKL_DSS_SUCCESS:
                raise RuntimeError('mkl.dss_factor_complex failed')
            self.__real = False
        else:
            ptr_as = _array_ptr(a_s)
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
            ptr_b = _array_ptr(b)
        else:
            b_s = b/self.__scale
            ptr_b = _array_ptr(b_s)
        ptr_x = _array_ptr(x)
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
        ptr_inertia = _array_ptr(inertia)
        err = self.__dss_statistics(ctypes.byref(self.__handle), \
                             ctypes.byref(opt), \
                             ptr_stat, ptr_inertia)
        if err is not self.__MKL_DSS_SUCCESS:
            raise RuntimeError('mkl.dss_statistics failed')
        return inertia

