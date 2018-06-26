# -*- coding: utf-8 -*-
"""
MKL cblas implementation of ndarray-vectors algebra.

Created on Thu Jun 14 12:34:53 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes
import numpy
import sys
sys.path.append('../..')

from raleigh.ndarray.mkl import Cblas

from raleigh.ndarray.algebra_bc import NDArrayVectors, NDArrayMatrix

def array_ptr(array, shift = 0):
    return ctypes.c_void_p(array.ctypes.data + shift)

def conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a

class Vectors(NDArrayVectors):
    def __init__(self, arg, nvec = 0, data_type = None):
        super(Vectors, self).__init__(arg, nvec, data_type)
        dt = self.data_type()
        self.__cblas = Cblas(dt)
    def __complex(self):
        dt = self.data_type()
        if dt == numpy.complex64:
            floats = ctypes.c_float * 2
        elif dt == numpy.complex128:
            floats = ctypes.c_double * 2
        else:
            raise ValueError('wrong data type %s passed to __complex' % repr(dt))
        return floats()
    def __to_float(self, v):
        dt = self.data_type()
        if dt == numpy.float32:
            s = ctypes.c_float(v)
        elif dt == numpy.float64:
            s = ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            s = self.__complex()
            s[0] = v.real
            s[1] = v.imag
        else:
            raise ValueError('data type %s not supported' % repr(dt))
        return s
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
            ptr_u = array_ptr(self.all_data(), i*vsize)
            ptr_v = array_ptr(other.all_data(), j*vsize)
            self.__cblas.copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
        else:
            mkl_n = ctypes.c_int(vdim)
            l = len(ind)
            for k in range(l):
                ptr_u = array_ptr(self.all_data(), int(ind[k])*vsize)
                ptr_v = array_ptr(other.all_data(), (j + k)*vsize)
                self.__cblas.copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)
    def scale(self, s):
        f, n = self.selected()
        vdim = self.dimension()
        mkl_n = ctypes.c_int(vdim)
        mkl_inc = ctypes.c_int(1)
        vsize = self.__cblas.dsize * vdim
        for i in range(n):
            if s[i] != 0.0:
                ptr_u = array_ptr(self.all_data(), (f + i)*vsize)
                mkl_s = self.__to_float(1.0/s[i])
                self.__cblas.scal(mkl_n, mkl_s, ptr_u, mkl_inc)
    def dots(self, other):
        n = self.nvec()
        v = numpy.ndarray((n,), dtype = self.data_type())
        vdim = self.dimension()
        mkl_n = ctypes.c_int(vdim)
        mkl_inc = ctypes.c_int(1)
        vsize = self.__cblas.dsize * vdim
        iu = self.selected()[0]
        iv = other.selected()[0]
        for i in range(n):
            ptr_u = array_ptr(self.all_data(), (iu + i)*vsize)
            ptr_v = array_ptr(other.all_data(), (iv + i)*vsize)
            if self.is_complex():
                s = self.__complex()
                self.__cblas.inner \
                    (mkl_n, ptr_v, mkl_inc, ptr_u, mkl_inc, s)
                v[i] = s[0] + 1j * s[1]
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
        ptr_u = array_ptr(other.data())
        ptr_v = array_ptr(self.data())
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
        ptr_u = array_ptr(output.data())
        ptr_v = array_ptr(self.data())
        ptr_q = ctypes.c_void_p(q.ctypes.data)
        if q.flags['C_CONTIGUOUS']:
            Trans = Cblas.Trans
            ldq = mkl_m
        elif q.flags['F_CONTIGUOUS']:
            Trans = Cblas.NoTrans
            ldq = mkl_k
        else:
            print('using non-optimized dot')
            output.data()[:,:] = numpy.dot(q.T, self.data())
            return
        self.__cblas.gemm(Cblas.ColMajor, Cblas.NoTrans, Trans, \
            mkl_n, mkl_m, mkl_k, \
            self.__cblas.mkl_one, ptr_v, mkl_n, ptr_q, ldq, \
            self.__cblas.mkl_zero, ptr_u, mkl_n)
    def apply(self, A, output, transp = False):
        a = A.data()
        if transp:
            is_complex = (a.dtype.kind == 'c')
            if is_complex:
                numpy.conj(self.data(), out = self.data())
            self.__apply(a.T, output)
            if is_complex:
                numpy.conj(output.data(), out = output.data())
                numpy.conj(self.data(), out = self.data())
        else:
            self.__apply(a, output)
    def __apply(self, q, output):
        f, s = output.selected();
        m = q.shape[0]
        n = self.dimension()
        k = self.nvec()
        mkl_n = ctypes.c_int(n)
        mkl_m = ctypes.c_int(m)
        mkl_k = ctypes.c_int(k)
        if q.flags['C_CONTIGUOUS']:
            Trans = Cblas.Trans
            ldq = mkl_n
        elif q.flags['F_CONTIGUOUS']:
            Trans = Cblas.NoTrans
            ldq = mkl_m
        else:
            print('using non-optimized dot')
            output.data()[:,:] = numpy.dot(self.data(), q.T)
            return
        ptr_u = array_ptr(output.data())
        ptr_v = array_ptr(self.data())
        ptr_q = ctypes.c_void_p(q.ctypes.data)
        self.__cblas.gemm(Cblas.ColMajor, Trans, Cblas.NoTrans, \
            mkl_m, mkl_k, mkl_n, \
            self.__cblas.mkl_one, ptr_q, ldq, ptr_v, mkl_n, \
            self.__cblas.mkl_zero, ptr_u, mkl_m)
    def add(self, other, s, q = None):
        f, m = self.selected();
        n = self.dimension()
        m = other.nvec()
        mkl_n = ctypes.c_int(n)
        mkl_m = ctypes.c_int(m)
        vsize = self.__cblas.dsize * n
        ptr_u = array_ptr(other.data())
        ptr_v = array_ptr(self.data())
        if numpy.isscalar(s):
            mkl_s = self.__to_float(s)
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
                    ldq = mkl_k
                elif q.flags['F_CONTIGUOUS']:
                    Trans = Cblas.NoTrans
                    ldq = mkl_m
                else:
                    print('using non-optimized dot')
                    self.data()[:,:] += s*numpy.dot(q.T, other.data())
                    return
                self.__cblas.gemm(Cblas.ColMajor, Cblas.NoTrans, Trans, \
                    mkl_n, mkl_k, mkl_m, \
                    mkl_s, ptr_u, mkl_n, ptr_q, ldq, \
                    self.__cblas.mkl_one, ptr_v, mkl_n)
        else:
            for i in range(m):
                ptr_u = array_ptr(other.data(), i*vsize)
                ptr_v = array_ptr(self.data(), i*vsize)
                mkl_inc = ctypes.c_int(1)
                mkl_s = self.__to_float(s[i])
                self.__cblas.axpy \
                    (mkl_n, mkl_s, ptr_u, mkl_inc, ptr_v, mkl_inc)

class Matrix(NDArrayMatrix):
    pass
