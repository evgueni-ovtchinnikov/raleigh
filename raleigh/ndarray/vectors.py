'''
Implementation of the abstract class Vectors based on numpy.ndarray

'''

import ctypes
import numbers
import numpy
import numpy.linalg as nla

from sys import platform

#print(platform)

try:
    if platform == 'win32':
        mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
    else:
        mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)
    HAVE_MKL = True
    CblasColMajor = 102
    CblasNoTrans = 111
    CblasTrans = 112
    CblasConjTrans = 113
    print('Using %d MKL threads' % mkl.mkl_get_max_threads())

except:
    HAVE_MKL = False

class Vectors:
    def __init__(self, arg, nvec = 0, data_type = None, with_mkl = True):
        if isinstance(arg, Vectors):
            self.__data = arg.__data.copy()
        elif isinstance(arg, numpy.ndarray):
            self.__data = arg
        elif isinstance(arg, numbers.Number):
            if data_type is None: # use default data type
                self.__data = numpy.zeros((nvec, arg))
            else:
                self.__data = numpy.zeros((nvec, arg), dtype = data_type)
        else:
            raise ValueError \
            ('wrong argument %s in constructor' % repr(type(arg)))
        self.__with_mkl = HAVE_MKL and with_mkl
        if self.__with_mkl:
            dt = self.__data.dtype
            if dt == numpy.float32:
                self.__dsize = 4
                self.__gemm = mkl.cblas_sgemm
                self.__axpy = mkl.cblas_saxpy
                self.__copy = mkl.cblas_scopy
                self.__scal = mkl.cblas_sscal
                self.__norm = mkl.cblas_snrm2
                self.__norm.restype = ctypes.c_float
                self.__inner = mkl.cblas_sdot
                self.__inner.restype = ctypes.c_float
                self.__mkl_one = ctypes.c_float(1.0)
                self.__mkl_zero = ctypes.c_float(0.0)
            elif dt == numpy.float64:
                self.__dsize = 8
                self.__gemm = mkl.cblas_dgemm
                self.__axpy = mkl.cblas_daxpy
                self.__copy = mkl.cblas_dcopy
                self.__scal = mkl.cblas_dscal
                self.__norm = mkl.cblas_dnrm2
                self.__norm.restype = ctypes.c_double
                self.__inner = mkl.cblas_ddot
                self.__inner.restype = ctypes.c_double
                self.__mkl_one = ctypes.c_double(1.0)
                self.__mkl_zero = ctypes.c_double(0.0)
            elif dt == numpy.complex64:
                self.__dsize = 8
                self.__gemm = mkl.cblas_cgemm
                self.__axpy = mkl.cblas_caxpy
                self.__copy = mkl.cblas_ccopy
                self.__scal = mkl.cblas_cscal
                self.__norm = mkl.cblas_scnrm2
                self.__norm.restype = ctypes.c_float
                self.__inner = mkl.cblas_cdotc_sub
                self.__cmplx_val = numpy.zeros((2,), dtype = numpy.float32)
                self.__cmplx_one = numpy.zeros((2,), dtype = numpy.float32)
                self.__cmplx_one[0] = 1.0
                self.__cmplx_zero = numpy.zeros((2,), dtype = numpy.float32)
                self.__mkl_one = ctypes.c_void_p(self.__cmplx_one.ctypes.data)
                self.__mkl_zero = ctypes.c_void_p(self.__cmplx_zero.ctypes.data)
            elif dt == numpy.complex128:
                self.__dsize = 16
                self.__gemm = mkl.cblas_zgemm
                self.__axpy = mkl.cblas_zaxpy
                self.__copy = mkl.cblas_zcopy
                self.__scal = mkl.cblas_zscal
                self.__norm = mkl.cblas_dznrm2
                self.__norm.restype = ctypes.c_double
                self.__inner = mkl.cblas_zdotc_sub
                self.__cmplx_val = numpy.zeros((2,), dtype = numpy.float64)
                self.__cmplx_one = numpy.zeros((2,), dtype = numpy.float64)
                self.__cmplx_one[0] = 1.0
                self.__cmplx_zero = numpy.zeros((2,), dtype = numpy.float64)
                self.__mkl_one = ctypes.c_void_p(self.__cmplx_one.ctypes.data)
                self.__mkl_zero = ctypes.c_void_p(self.__cmplx_zero.ctypes.data)
            else:
                raise ValueError('data type %s not supported' % repr(dt))
        m, n = self.__data.shape
        self.__selected = (0, m)
    def __to_mkl_float(self, v):
        dt = self.__data.dtype
        if dt == numpy.float32:
            return ctypes.c_float(v)
        elif dt == numpy.float64:
            return ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            self.__cmplx_val[0] = v.real
            self.__cmplx_val[1] = v.imag
            return ctypes.c_void_p(self.__cmplx_val.ctypes.data)
        else:
            raise ValueError('data type %s not supported' % repr(dt))
    def dimension(self):
        return self.__data.shape[1]
    def nvec(self):
        return self.__selected[1]
    def selected(self):
        return self.__selected
    def select(self, nv, first = 0):
        assert nv <= self.__data.shape[0] and first >= 0
        self.__selected = (first, nv)
    def select_all(self):
        self.select(self.__data.shape[0])
    def data_type(self):
        return self.__data.dtype
    def is_complex(self):
        v = self.__data[0,0]
        return type(v) is numpy.complex64 or type(v) is numpy.complex128
##        return numpy.iscomplex(self.__data[0,0])
##        return isinstance(self.__data[0,0], complex)
    def clone(self):
        return Vectors(self) #.__data.copy())
    def new_vectors(self, nv = 0):
        m, n = self.__data.shape
        return Vectors(n, nv, self.__data.dtype)
##        data = numpy.ones((nv, n), dtype = self.__data.dtype)
##        return Vectors(data)
    def zero(self):
        f, n = self.__selected;
        self.__data[f : f + n, :] = 0.0
    def fill(self, array_or_value):
        f, n = self.__selected;
        self.__data[f : f + n, :] = array_or_value
    def fill_random(self):
        iv, nv = self.__selected
        m, n = self.__data.shape
        #data = numpy.zeros((nv, n), dtype = self.__data.dtype)
        self.__data[iv : iv + nv,:] = 2*numpy.random.rand(nv, n) - 1
        #return Vectors(data)
    def fill_orthogonal(self, m):
        iv, nv = self.__selected
        k, n = self.__data.shape
        if n < m:
            print('Warning: number of vectors too large, reducing')
            m = n
        #a = numpy.zeros((m, n), dtype = self.__data.dtype)
        a = self.__data[iv : iv + nv, :]
        a[0,0] = 1.0
        i = 1
        while 2*i < m:
            a[i : 2*i, :i] = a[:i, :i]
            a[:i, i : 2*i] = a[:i, :i]
            a[i : 2*i, i : 2*i] = -a[:i, :i]
            i *= 2
        k = i
        j = 2*i
        if j > n:
            for i in range(k, m):
                a[i, i] = 1.0
            return Vectors(a)
        while j <= n:
            a[:k, i : j] = a[:k, :i]
            i, j = j, 2*j
        j = i//2
        a[k : m,   : j] = a[:(m - k), : j]
        a[k : m, j : i] = -a[:(m - k), j : i]
        #return Vectors(a)
    def data(self, i = None):
        f, n = self.__selected
        if i is None:
            return self.__data[f : f + n, :]
        else:
            return self.__data[f + i, :]
    def append(self, other):
        self.__data = numpy.concatenate((self.__data, other.data()))
        self.select_all()

    # BLAS level 1
    def copy(self, other, ind = None):
        i, n = self.__selected
        j, m = other.__selected
        if self.__with_mkl:
            vdim = self.dimension()
            mkl_inc = ctypes.c_int(1)
            vsize = self.__dsize * vdim
            if ind is None:
                mkl_n = ctypes.c_int(n*vdim)
                data_u = self.__data.ctypes.data + i*vsize
                data_v = other.__data.ctypes.data + j*vsize
                ptr_u = ctypes.c_void_p(data_u)
                ptr_v = ctypes.c_void_p(data_v)
                self.__copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
            else:
                mkl_n = ctypes.c_int(vdim)
                l = len(ind)
#                print(type(self.__data.ctypes.data))
#                print(type(ind[0]))
#                print(type(vsize))
                for k in range(l):
                    data_u = self.__data.ctypes.data + int(ind[k])*vsize
                    data_v = other.__data.ctypes.data + (j + k)*vsize
#                    print(type(data_u))
                    ptr_u = ctypes.c_void_p(data_u)
                    ptr_v = ctypes.c_void_p(data_v)
                    self.__copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
        else:
            if ind is None:
                other.__data[j : j + n, :] = self.__data[i : i + n, :]
            else:
                other.__data[j : j + len(ind), :] = self.__data[ind, :]
    def scale(self, s):
        f, n = self.__selected;
        if self.__with_mkl:
            vdim = self.dimension()
            mkl_n = ctypes.c_int(vdim)
            mkl_inc = ctypes.c_int(1)
            vsize = self.__dsize * vdim
            for i in range(n):
                if s[i] != 0.0:
                    data_u = self.__data.ctypes.data + (f + i)*vsize
                    ptr_u = ctypes.c_void_p(data_u)
                    mkl_s = self.__to_mkl_float(1.0/s[i])
                    self.__scal(mkl_n, mkl_s, ptr_u, mkl_inc)
        else:
            for i in range(n):
                if s[i] != 0.0:
                    self.__data[f + i, :] /= s[i]
    def dots(self, other):
        n = self.__selected[1]
        v = numpy.ndarray((n,), dtype = self.__data.dtype)
        if self.__with_mkl:
            vdim = self.dimension()
            mkl_n = ctypes.c_int(vdim)
            mkl_inc = ctypes.c_int(1)
            vsize = self.__dsize * vdim
            if self.is_complex():
                ptr_r = ctypes.c_void_p(self.__cmplx_val.ctypes.data)
            for i in range(n):
                iu = self.__selected[0]
                iv = other.__selected[0]
                data_u = self.__data.ctypes.data + (iu + i)*vsize
                data_v = other.__data.ctypes.data + (iv + i)*vsize
                ptr_u = ctypes.c_void_p(data_u)
                ptr_v = ctypes.c_void_p(data_v)
                if self.is_complex():
                    self.__inner(mkl_n, ptr_v, mkl_inc, ptr_u, mkl_inc, ptr_r)
                    res = self.__cmplx_val
                    v[i] = res[0] + 1j * res[1]
                else:
                    v[i] = self.__inner(mkl_n, ptr_v, mkl_inc, ptr_u, mkl_inc)
        else:
            for i in range(n):
                if other.is_complex():
                    s =  numpy.dot(other.data(i).conj(), self.data(i))
                else:
                    s = numpy.dot(other.data(i), self.data(i))
                v[i] = s
        return v

    # BLAS level 3
    def dot(self, other):
        if self.__with_mkl:
            m, n = self.__data.shape
            m = self.nvec()
            k = other.nvec()
            q = numpy.ndarray((m, k), dtype = self.__data.dtype)
            mkl_n = ctypes.c_int(n)
            mkl_m = ctypes.c_int(m)
            mkl_k = ctypes.c_int(k)
            vsize = self.__dsize * n
            data_u = other.__data.ctypes.data + other.__selected[0] * vsize
            data_v = self.__data.ctypes.data + self.__selected[0] * vsize
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            ptr_q = ctypes.c_void_p(q.ctypes.data)
            if self.is_complex():
                Trans = CblasConjTrans
#                t = numpy.dot(other.data().conj(), self.data().T)
            else:
                Trans = CblasTrans
#                t = numpy.dot(other.data(), self.data().T)
            self.__gemm(CblasColMajor, Trans, CblasNoTrans, \
                mkl_k, mkl_m, mkl_n, \
                self.__mkl_one, ptr_u, mkl_n, ptr_v, mkl_n, \
                self.__mkl_zero, ptr_q, mkl_k)
#            print(nla.norm(t - q.T))
            return q.T
        else:
            if other.is_complex():
                return numpy.dot(other.data().conj(), self.data().T)
            else:
                return numpy.dot(other.data(), self.data().T)
    def multiply(self, q, output):
        f, n = output.__selected;
        m = q.shape[1]
        if self.__with_mkl:
            n = self.dimension()
            fs = self.__selected[0]
            mkl_n = ctypes.c_int(n)
            mkl_m = ctypes.c_int(m)
            mkl_k = ctypes.c_int(self.nvec())
            vsize = self.__dsize * n
            data_u = output.__data.ctypes.data + f * vsize
            data_v = self.__data.ctypes.data + fs * vsize
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            ptr_q = ctypes.c_void_p(q.ctypes.data)
            self.__gemm(CblasColMajor, CblasNoTrans, CblasTrans, \
                mkl_n, mkl_m, mkl_k, \
                self.__mkl_one, ptr_v, mkl_n, ptr_q, mkl_m, \
                self.__mkl_zero, ptr_u, mkl_n)
        else:
            if output.__data[f : f + m, :].flags['C_CONTIGUOUS']:
                #print('using optimized dot')
                numpy.dot(q.T, self.data(), out = output.__data[f : f + m, :])
                return
            print('using non-optimized dot')
            output.__data[f : f + m, :] = numpy.dot(q.T, self.data())
    def add(self, other, s, q = None):
        f, m = self.__selected;
        if self.__with_mkl:
            n = self.dimension()
            fu, m = other.__selected
            mkl_n = ctypes.c_int(n)
            mkl_m = ctypes.c_int(m)
            vsize = self.__dsize * n
            data_u = other.__data.ctypes.data + fu * vsize
            data_v = self.__data.ctypes.data + f * vsize
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            if numpy.isscalar(s):
                mkl_s = self.__to_mkl_float(s)
                if q is None:
                    mkl_nm = ctypes.c_int(n*m)
                    mkl_inc = ctypes.c_int(1)
                    self.__axpy(mkl_nm, mkl_s, ptr_u, mkl_inc, ptr_v, mkl_inc)
                else:
                    mkl_k = ctypes.c_int(q.shape[1])
                    ptr_q = ctypes.c_void_p(q.ctypes.data)
                    self.__gemm(CblasColMajor, CblasNoTrans, CblasTrans, \
                        mkl_n, mkl_k, mkl_m, \
                        mkl_s, ptr_u, mkl_n, ptr_q, mkl_k, \
                        self.__mkl_one, ptr_v, mkl_n)
            else:
                for i in range(m):
                    data_u = other.__data.ctypes.data + (fu + i) * vsize
                    data_v = self.__data.ctypes.data + (f + i) * vsize
                    ptr_u = ctypes.c_void_p(data_u)
                    ptr_v = ctypes.c_void_p(data_v)
                    mkl_inc = ctypes.c_int(1)
                    mkl_s = self.__to_mkl_float(s[i])
                    self.__axpy(mkl_n, mkl_s, ptr_u, mkl_inc, ptr_v, mkl_inc)
        else:
            if numpy.isscalar(s):
                if q is None:
                    self.__data[f : f + m, :] += s*other.data()
                else:
                    self.__data[f : f + m, :] += s*numpy.dot(q.T, other.data())
                return
            for i in range(m):
                self.__data[f + i, :] += s[i]*other.data(i)

