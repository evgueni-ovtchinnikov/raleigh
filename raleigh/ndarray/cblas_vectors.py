# -*- coding: utf-8 -*-
"""
MKL cblas implementation of ndarray-vectors algebra.

Created on Thu Jun 14 12:34:53 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes
import numpy

from raleigh.ndarray.mkl import Cblas

from raleigh.ndarray.ndarray_vectors import NDArrayVectors

def conjugate(a):
    if isinstance(a[0,0], complex):
        return a.conj()
    else:
        return a

class Vectors(NDArrayVectors):
    def __init__(self, arg, nvec = 0, data_type = None):
        super(Vectors, self).__init__(arg, nvec, data_type)
        dt = self.data_type()
        self.__cblas = Cblas(dt)
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
    def new_vectors(self, nv = 0):
        return Vectors(self.dimension(), nv, self.data_type())
    def clone(self):
        return Vectors(self)

    # BLAS level 1
    def copy(self, other, ind = None):
        i, n = self.selected()
        j, m = other.selected()
        vdim = self.dimension()
        mkl_inc = ctypes.c_int(1)
        vsize = self.__cblas.dsize * vdim
        if ind is None:
            mkl_n = ctypes.c_int(n*vdim)
            data_u = self.all_data().ctypes.data + i*vsize
            data_v = other.all_data().ctypes.data + j*vsize
            ptr_u = ctypes.c_void_p(data_u)
            ptr_v = ctypes.c_void_p(data_v)
            self.__cblas.copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
        else:
            mkl_n = ctypes.c_int(vdim)
            l = len(ind)
            for k in range(l):
                data_u = self.all_data().ctypes.data + int(ind[k])*vsize
                data_v = other.all_data().ctypes.data + (j + k)*vsize
                ptr_u = ctypes.c_void_p(data_u)
                ptr_v = ctypes.c_void_p(data_v)
                self.__cblas.copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
    def scale(self, s):
        f, n = self.selected()
        vdim = self.dimension()
        mkl_n = ctypes.c_int(vdim)
        mkl_inc = ctypes.c_int(1)
        vsize = self.__cblas.dsize * vdim
        for i in range(n):
            if s[i] != 0.0:
                data_u = self.all_data().ctypes.data + (f + i)*vsize
                ptr_u = ctypes.c_void_p(data_u)
                mkl_s = self.__to_mkl_float(1.0/s[i])
                self.__cblas.scal(mkl_n, mkl_s, ptr_u, mkl_inc)
    def dots(self, other):
        n = self.nvec()
        v = numpy.ndarray((n,), dtype = self.data_type())
        vdim = self.dimension()
        mkl_n = ctypes.c_int(vdim)
        mkl_inc = ctypes.c_int(1)
        vsize = self.__cblas.dsize * vdim
        if self.is_complex():
            ptr_r = ctypes.c_void_p(self.__cblas.cmplx_val.ctypes.data)
        for i in range(n):
            iu = self.selected()[0]
            iv = other.selected()[0]
            data_u = self.all_data().ctypes.data + (iu + i)*vsize
            data_v = other.all_data().ctypes.data + (iv + i)*vsize
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
        return v

    # BLAS level 3
    def dot(self, other):
        m, n = self.all_data().shape
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
        else:
            Trans = Cblas.Trans
        self.__cblas.gemm(Cblas.ColMajor, Trans, Cblas.NoTrans, \
            mkl_m, mkl_k, mkl_n, \
            self.__cblas.mkl_one, ptr_v, mkl_n, ptr_u, mkl_n, \
            self.__cblas.mkl_zero, ptr_q, mkl_m)
        return conjugate(q)
    def multiply(self, q, output):
        f, s = output.selected();
        m = q.shape[1]
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
    def apply(self, q, output):
        if output.data().flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(self.data(), q.T, out = output.data())
        else:
            #print('using non-optimized dot')
            output.data()[:,:] = numpy.dot(self.data(), q.T)
    def add(self, other, s, q = None):
        f, m = self.selected();
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
