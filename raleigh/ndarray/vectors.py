'''
Implementation of the abstract class Vectors based on numpy.ndarray

'''

import ctypes
import numbers
import numpy

try:
    from raleigh.ndarray.mkl import Cblas
    HAVE_MKL = True
except:
    HAVE_MKL = False

def conjugate(a):
    if isinstance(a[0,0], complex):
        return a.conj()
    else:
        return a

class Vectors:
    def __init__(self, arg, nvec = 0, data_type = None, with_mkl = None):
        if isinstance(arg, Vectors):
            self.__data = arg.__data.copy()
            if with_mkl is None:
                with_mkl = arg.__with_mkl
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
        if with_mkl is None:
            with_mkl = True
        self.__with_mkl = HAVE_MKL and with_mkl
        if self.__with_mkl:
            dt = self.data_type()
            self.__cblas = Cblas(dt)
        m, n = self.__data.shape
        self.__selected = (0, m)
    def __to_mkl_float(self, v):
        dt = self.data_type()
        if dt == numpy.float32:
            return ctypes.c_float(v)
        elif dt == numpy.float64:
            return ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            self.__cblas.cmplx_val[0] = v.real
            self.__cblas.cmplx_val[1] = v.imag
            return ctypes.c_void_p(self.__cblas.cmplx_val.ctypes.data)
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
#        return type(self.__data[0,0])
        return self.__data.dtype.type
    def is_complex(self):
#        v = self.__data[0,0]
#        return type(v) is numpy.complex64 or type(v) is numpy.complex128
##        return numpy.iscomplex(self.__data[0,0])
        return isinstance(self.__data[0,0], complex)
    def clone(self):
        return Vectors(self)
    def new_vectors(self, nv = 0):
        m, n = self.__data.shape
        return Vectors(n, nv, self.data_type(), self.__with_mkl)
    def zero(self):
        f, n = self.__selected;
        self.__data[f : f + n, :] = 0.0
    def fill(self, array_or_value):
        f, n = self.__selected;
        self.__data[f : f + n, :] = array_or_value
    def fill_random(self):
        iv, nv = self.__selected
        m, n = self.__data.shape
        self.__data[iv : iv + nv,:] = 2*numpy.random.rand(nv, n) - 1
    def fill_orthogonal(self, m):
        iv, nv = self.__selected
        k, n = self.__data.shape
        if n < m:
            print('Warning: number of vectors too large, reducing')
            m = n
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
            vsize = self.__cblas.dsize * vdim
            if ind is None:
                mkl_n = ctypes.c_int(n*vdim)
                data_u = self.__data.ctypes.data + i*vsize
                data_v = other.__data.ctypes.data + j*vsize
                ptr_u = ctypes.c_void_p(data_u)
                ptr_v = ctypes.c_void_p(data_v)
                self.__cblas.copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
            else:
                mkl_n = ctypes.c_int(vdim)
                l = len(ind)
                for k in range(l):
                    data_u = self.__data.ctypes.data + int(ind[k])*vsize
                    data_v = other.__data.ctypes.data + (j + k)*vsize
                    ptr_u = ctypes.c_void_p(data_u)
                    ptr_v = ctypes.c_void_p(data_v)
                    self.__cblas.copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
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
            vsize = self.__cblas.dsize * vdim
            for i in range(n):
                if s[i] != 0.0:
                    data_u = self.__data.ctypes.data + (f + i)*vsize
                    ptr_u = ctypes.c_void_p(data_u)
                    mkl_s = self.__to_mkl_float(1.0/s[i])
                    self.__cblas.scal(mkl_n, mkl_s, ptr_u, mkl_inc)
        else:
            for i in range(n):
                if s[i] != 0.0:
                    self.__data[f + i, :] /= s[i]
    def dots(self, other):
        n = self.__selected[1]
        v = numpy.ndarray((n,), dtype = self.data_type())
        if self.__with_mkl:
            vdim = self.dimension()
            mkl_n = ctypes.c_int(vdim)
            mkl_inc = ctypes.c_int(1)
            vsize = self.__cblas.dsize * vdim
            if self.is_complex():
                ptr_r = ctypes.c_void_p(self.__cblas.cmplx_val.ctypes.data)
            for i in range(n):
                iu = self.__selected[0]
                iv = other.__selected[0]
                data_u = self.__data.ctypes.data + (iu + i)*vsize
                data_v = other.__data.ctypes.data + (iv + i)*vsize
                ptr_u = ctypes.c_void_p(data_u)
                ptr_v = ctypes.c_void_p(data_v)
                if self.is_complex():
                    self.__cblas.inner \
                        (mkl_n, ptr_v, mkl_inc, ptr_u, mkl_inc, ptr_r)
                    res = self.__cblas.cmplx_val
                    v[i] = res[0] + 1j * res[1]
                else:
                    v[i] = self.__cblas.inner \
                        (mkl_n, ptr_v, mkl_inc, ptr_u, mkl_inc)
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
            q = numpy.ndarray((k, m), dtype = self.data_type())
            mkl_n = ctypes.c_int(n)
            mkl_m = ctypes.c_int(m)
            mkl_k = ctypes.c_int(k)
            data_u = other.data().ctypes.data
            data_v = self.data().ctypes.data
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            ptr_q = ctypes.c_void_p(q.ctypes.data)
            if self.is_complex():
                Trans = Cblas.ConjTrans
#                t = numpy.dot(other.data().conj(), self.data().T)
            else:
                Trans = Cblas.Trans
#                t = numpy.dot(other.data(), self.data().T)
            self.__cblas.gemm(Cblas.ColMajor, Trans, Cblas.NoTrans, \
                mkl_m, mkl_k, mkl_n, \
                self.__cblas.mkl_one, ptr_v, mkl_n, ptr_u, mkl_n, \
                self.__cblas.mkl_zero, ptr_q, mkl_m)
            return conjugate(q)
        else:
            if other.is_complex():
                return numpy.dot(other.data().conj(), self.data().T)
            else:
                return numpy.dot(other.data(), self.data().T)
    def multiply(self, q, output):
        f, s = output.__selected;
        m = q.shape[1]
        if self.__with_mkl:
            n = self.dimension()
            mkl_n = ctypes.c_int(n)
            mkl_m = ctypes.c_int(m)
            mkl_k = ctypes.c_int(self.nvec())
            data_u = output.data().ctypes.data
            data_v = self.data().ctypes.data
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            ptr_q = ctypes.c_void_p(q.ctypes.data)
            if q.flags['C_CONTIGUOUS']:
                Trans = Cblas.Trans
            else:
                Trans = Cblas.NoTrans
            self.__cblas.gemm(Cblas.ColMajor, Cblas.NoTrans, Trans, \
                mkl_n, mkl_m, mkl_k, \
                self.__cblas.mkl_one, ptr_v, mkl_n, ptr_q, mkl_m, \
                self.__cblas.mkl_zero, ptr_u, mkl_n)
        else:
            if output.__data[f : f + m, :].flags['C_CONTIGUOUS']:
                #print('using optimized dot')
                numpy.dot(q.T, self.data(), out = output.__data[f : f + m, :])
                return
            #print('using non-optimized dot')
            output.__data[f : f + m, :] = numpy.dot(q.T, self.data())
    def add(self, other, s, q = None):
        f, m = self.__selected;
        if self.__with_mkl:
            n = self.dimension()
            m = other.nvec()
            mkl_n = ctypes.c_int(n)
            mkl_m = ctypes.c_int(m)
            vsize = self.__cblas.dsize * n
            data_u = other.data().ctypes.data
            data_v = self.data().ctypes.data
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            if numpy.isscalar(s):
                mkl_s = self.__to_mkl_float(s)
                if q is None:
                    mkl_nm = ctypes.c_int(n*m)
                    mkl_inc = ctypes.c_int(1)
                    self.__cblas.axpy \
                        (mkl_nm, mkl_s, ptr_u, mkl_inc, ptr_v, mkl_inc)
                else:
                    mkl_k = ctypes.c_int(q.shape[1])
                    ptr_q = ctypes.c_void_p(q.ctypes.data)
                    if q.flags['C_CONTIGUOUS']:
                        Trans = Cblas.Trans
                    else:
                        Trans = Cblas.NoTrans
                    self.__cblas.gemm(Cblas.ColMajor, Cblas.NoTrans, Trans, \
                        mkl_n, mkl_k, mkl_m, \
                        mkl_s, ptr_u, mkl_n, ptr_q, mkl_k, \
                        self.__cblas.mkl_one, ptr_v, mkl_n)
            else:
                for i in range(m):
                    ptr_u = ctypes.c_void_p(data_u)
                    ptr_v = ctypes.c_void_p(data_v)
                    mkl_inc = ctypes.c_int(1)
                    mkl_s = self.__to_mkl_float(s[i])
                    self.__cblas.axpy \
                        (mkl_n, mkl_s, ptr_u, mkl_inc, ptr_v, mkl_inc)
                    data_u += vsize
                    data_v += vsize
        else:
            if numpy.isscalar(s):
                if q is None:
                    self.__data[f : f + m, :] += s*other.data()
                else:
                    self.__data[f : f + m, :] += s*numpy.dot(q.T, other.data())
                return
            for i in range(m):
                self.__data[f + i, :] += s[i]*other.data(i)

