# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:19:47 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes
import numbers
import numpy
import sys
sys.path.append('..')

import raleigh.cuda.cuda as cuda

from raleigh.cuda.cublas import Cublas

def try_calling(err):
    if err != 0:
        raise RuntimeError('cuda error')

def shifted_ptr(dev_ptr, shift = 0):
#    print(type(dev_ptr), type(shift))
    ptr = ctypes.cast(dev_ptr, ctypes.c_void_p)
    return ctypes.cast(ptr.value + shift, ctypes.POINTER(ctypes.c_ubyte))

def conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a

class Vectors:
    def __init__(self, arg, nvec = 0, dtype = None):
        self.__data = ctypes.POINTER(ctypes.c_ubyte)()
        if dtype is None:
            dtype = numpy.float64
            self.__is_complex = False
        if dtype == numpy.float32:
            dsize = 4
            self.__is_complex = False
        elif dtype == numpy.float64:
            dsize = 8
            self.__is_complex = False
        elif dtype == numpy.complex64:
            dsize = 8
            self.__is_complex = True
        elif dtype == numpy.complex128:
            dsize = 16
            self.__is_complex = True
        else:
            raise ValueError('data type %s not supported' % repr(dtype))
        if isinstance(arg, Vectors):
            n = arg.dimension()
            m = arg.nvec()
            self.__is_complex = arg.__is_complex
            dtype = arg.__dtype
            dsize = arg.__dsize
            size = n*m*dsize
            try_calling(cuda.malloc(ctypes.byref(self.__data), size))
            try_calling(cuda.memcpy(self.__data, arg.__data, size, cuda.memcpyD2D))
        elif isinstance(arg, numpy.ndarray):
            m, n = arg.shape
            dtype = arg.dtype.type
            dsize = arg.itemsize
            size = n*m*dsize
            try_calling(cuda.malloc(ctypes.byref(self.__data), size))
            ptr = ctypes.c_void_p(arg.ctypes.data)
            try_calling(cuda.memcpy(self.__data, ptr, size, cuda.memcpyH2D))
            self.__is_complex = \
                (dtype == numpy.complex64 or dtype == numpy.complex128)
#            self.__is_complex = isinstance(arg[0,0], complex) # lying!!!
        elif isinstance(arg, numbers.Number):
            n = arg
            m = nvec
            size = n*m*dsize
            try_calling(cuda.malloc(ctypes.byref(self.__data), size))
            try_calling(cuda.memset(self.__data, 0, size))
        else:
            raise ValueError \
                ('wrong argument %s in constructor' % repr(type(arg)))
        self.__selected = (0, m)
        self.__vdim = n
        self.__nvec = m
        self.__dsize = dsize
        self.__dtype = dtype
        self.__cublas = Cublas(dtype)
    def __del__(self):
        try_calling(cuda.free(self.__data))
    def __float(self):
        dt = self.data_type()
        if dt == numpy.float32:
            return ctypes.c_float()
        elif dt == numpy.float64:
            return ctypes.c_double()
        else:
            raise ValueError('wrong data type %s passed to __float' % repr(dt))
    def __floats(self):
        dt = self.data_type()
        if dt == numpy.float32:
            floats = ctypes.c_float * 1
        elif dt == numpy.float64:
            floats = ctypes.c_double * 1
        elif dt == numpy.complex64:
            floats = ctypes.c_float * 2
        elif dt == numpy.complex128:
            floats = ctypes.c_double * 2
        else:
            raise ValueError('wrong data type %s passed to __complex' % repr(dt))
        return floats()
    def __to_floats(self, v):
        s = self.__floats()
        dt = self.data_type()
        if dt == numpy.float32:
            s[0] = ctypes.c_float(v)
        elif dt == numpy.float64:
            s[0] = ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            s[0] = v.real
            s[1] = v.imag
        else:
            raise ValueError('data type %s not supported' % repr(dt))
        return s
    def new_vectors(self, nv = 0):
        return Vectors(self.dimension(), nv, self.data_type())
    def clone(self):
        return Vectors(self)
    def cublas(self):
        return self.__cublas
    def cublas_handle(self):
        return self.__cublas.handle

    def dimension(self):
        return self.__vdim
    def first(self):
        return self.__selected[0]
    def nvec(self):
        return self.__selected[1]
    def selected(self):
        return self.__selected
    def select(self, nv, first = 0):
        assert nv <= self.__nvec and first >= 0
        self.__selected = (first, nv)
    def select_all(self):
        self.select(self.__nvec)
    def data_type(self):
        return self.__dtype
    def is_complex(self):
        return self.__is_complex

    def all_data_ptr(self):
        return self.__data
    def data_ptr(self):
        return shifted_ptr(self.__data, self.first())

    # BLAS level 1
    def copy(self, other, ind = None):
        i, m = self.selected()
        j, l = other.selected()
        vdim = self.dimension()
        inc = ctypes.c_int(1)
        vsize = self.__cublas.dsize * vdim
        data_u = self.all_data_ptr()
        data_v = other.all_data_ptr()
        if ind is None:
            n = ctypes.c_int(m*vdim)
            ptr_u = shifted_ptr(data_u, i*vsize)
            ptr_v = shifted_ptr(data_v, j*vsize)
            self.__cublas.copy(self.__cublas.handle, n, ptr_u, inc, ptr_v, inc)
        else:
            n = ctypes.c_int(vdim)
            l = len(ind)
            for k in range(l):
                ptr_u = shifted_ptr(data_u, int(ind[k])*vsize)
                ptr_v = shifted_ptr(data_v, (j + k)*vsize)
                self.__cublas.copy \
                    (self.__cublas.handle, n, ptr_u, inc, ptr_v, inc)
    def scale(self, s):
        f, m = self.selected()
        vdim = self.dimension()
        n = ctypes.c_int(vdim)
        inc = ctypes.c_int(1)
        vsize = self.__cublas.dsize * vdim
        data_u = self.all_data_ptr()
        for i in range(m):
            ptr_u = shifted_ptr(data_u, (f + i)*vsize)
            if s[i] != 0.0:
                r = self.__to_floats(1.0/s[i])
                self.__cublas.scal(self.__cublas.handle, n, r, ptr_u, inc)
    def dots(self, other):
        m = self.nvec()
        v = numpy.ndarray((m,), dtype = self.data_type())
        vdim = self.dimension()
        n = ctypes.c_int(vdim)
        inc = ctypes.c_int(1)
        vsize = self.__dsize * vdim
        data_u = self.all_data_ptr()
        data_v = other.all_data_ptr()
        iu = self.first()
        iv = other.first()
        for i in range(m):
            ptr_u = shifted_ptr(data_u, (iu + i)*vsize)
            ptr_v = shifted_ptr(data_v, (iv + i)*vsize)
            s = self.__floats()
            self.__cublas.dot \
                (self.__cublas.handle, n, ptr_v, inc, ptr_u, inc, s)
            if self.is_complex():
                v[i] = s[0] + 1j * s[1]
            else:
                v[i] = s[0]
        return v

    # BLAS level 3
    def dot(self, other):
        m = self.nvec()
        n = self.dimension()
        k = other.nvec()
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        c_k = ctypes.c_int(k)
        dptr_u = other.data_ptr()
        dptr_v = self.data_ptr()
        dptr_q = ctypes.POINTER(ctypes.c_ubyte)()
        size_q = k*m*self.__dsize
        try_calling(cuda.malloc(ctypes.byref(dptr_q), size_q))
        if self.is_complex():
            Trans = Cublas.ConjTrans
        else:
            Trans = Cublas.Trans
        one = self.__to_floats(1.0)
        zero = self.__to_floats(0.0)
        self.__cublas.gemm(self.__cublas.handle, Trans, Cublas.NoTrans, \
            c_m, c_k, c_n, one, dptr_v, c_n, dptr_u, c_n, zero, dptr_q, c_m)
        q = numpy.ndarray((k, m), dtype = self.data_type())
        hptr_q = ctypes.c_void_p(q.ctypes.data)
        try_calling(cuda.memcpy(hptr_q, dptr_q, size_q, cuda.memcpyD2H))
        try_calling(cuda.free(dptr_q))
        return conjugate(q)
    def multiply(self, a, output):
        m = a.shape[1]
        n = self.dimension()
        k = self.nvec()
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        c_k = ctypes.c_int(k)
        dptr_u = output.data_ptr()
        dptr_v = self.data_ptr()
        dptr_q = ctypes.POINTER(ctypes.c_ubyte)()
        size_q = k*m*self.__dsize
        try_calling(cuda.malloc(ctypes.byref(dptr_q), size_q))
        if a.flags['C_CONTIGUOUS'] or a.flags['F_CONTIGUOUS']:
            q = a
        else:
            q = a.copy()
        hptr_q = ctypes.c_void_p(q.ctypes.data)
        try_calling(cuda.memcpy(dptr_q, hptr_q, size_q, cuda.memcpyH2D))
        if q.flags['C_CONTIGUOUS']:
            Trans = Cublas.Trans
            ldq = c_m
        elif q.flags['F_CONTIGUOUS']:
            Trans = Cublas.NoTrans
            ldq = c_k
        one = self.__to_floats(1.0)
        zero = self.__to_floats(0.0)
        self.__cublas.gemm(self.__cublas.handle, Cublas.NoTrans, Trans, \
            c_n, c_m, c_k, one, dptr_v, c_n, dptr_q, ldq, zero, dptr_u, c_n)
        try_calling(cuda.free(dptr_q))

class Matrix:
    def __init__(self, array):
        self.__data = ctypes.POINTER(ctypes.c_ubyte)()
        self.__shape = array.shape
        self.__dtype = array.dtype.type
        self.__is_complex = (array.dtype.kind == 'c')
        if self.__is_complex and not array.flags['F_CONTIGUOUS']:
            print('copyint to F-contiguous array...')
            a = numpy.ndarray(array.shape, dtype = array.dtype, order = 'F')
            a[:,:] = array.copy()
        else:
            a = array
        self.__flags = a.flags
        m, n = a.shape
        dsize = a.itemsize
        size = n*m*dsize
        try_calling(cuda.malloc(ctypes.byref(self.__data), size))
        ptr = ctypes.c_void_p(a.ctypes.data)
        try_calling(cuda.memcpy(self.__data, ptr, size, cuda.memcpyH2D))
    def __floats(self):
        dt = self.__dtype
        if dt == numpy.float32:
            floats = ctypes.c_float * 1
        elif dt == numpy.float64:
            floats = ctypes.c_double * 1
        elif dt == numpy.complex64:
            floats = ctypes.c_float * 2
        elif dt == numpy.complex128:
            floats = ctypes.c_double * 2
        else:
            raise ValueError('wrong data type %s passed to __complex' % repr(dt))
        return floats()
    def __to_floats(self, v):
        s = self.__floats()
        dt = self.__dtype
        if dt == numpy.float32:
            s[0] = ctypes.c_float(v)
        elif dt == numpy.float64:
            s[0] = ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            s[0] = v.real
            s[1] = v.imag
        else:
            raise ValueError('data type %s not supported' % repr(dt))
        return s
    def data_ptr(self):
        return self.__data
    def shape(self):
        return self.__shape
    def is_complex(self):
        return self.__is_complex

    def apply(self, x, y, transp = False):
        if x.data_type() != self.__dtype or y.data_type() != self.__dtype:
            raise ValueError('Matrix and vectors data types differ')
        if transp:
            n, m = self.__shape
            if n != y.dimension() or m != x.dimension():
                raise ValueError('Matrix and vectors dimensions incompatible')
        else:
            m, n = self.__shape
            if m != y.dimension() or n != x.dimension():
                raise ValueError('Matrix and vectors dimensions incompatible')
        k = x.nvec()
        if k != y.nvec():
            raise ValueError('Numbers of input and output vectors differ')
        k = ctypes.c_int(k)
        nx = ctypes.c_int(x.dimension())
        ny = ctypes.c_int(y.dimension())
        if self.__flags['C_CONTIGUOUS']:
            if transp:
                Trans = Cublas.NoTrans
            else:
                if self.__is_complex:
                    raise ValueError('Complex matrix must be F-contiguous')
                else:
                    Trans = Cublas.Trans
            lda = ctypes.c_int(n)
        elif self.__flags['F_CONTIGUOUS']:
            if transp:
                if self.__is_complex:
                    Trans = Cublas.ConjTrans
                else:
                    Trans = Cublas.Trans
            else:
                Trans = Cublas.NoTrans
            lda = ctypes.c_int(m)
        one = self.__to_floats(1.0)
        zero = self.__to_floats(0.0)
        dptr_a = self.__data
        dptr_x = x.data_ptr()
        dptr_y = y.data_ptr()
        x.cublas().gemm(x.cublas_handle(), Trans, Cublas.NoTrans, \
            ny, k, nx, one, dptr_a, lda, dptr_x, nx, zero, dptr_y, ny)
