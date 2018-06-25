# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:19:47 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes
import numbers
import numpy

import raleigh.cuda.cuda as cuda

from raleigh.cuda.cublas import Cublas

def try_calling(err):
    if err != 0:
        raise RuntimeError('cuda error')

def shifted_ptr(dev_ptr, shift):
#    print(type(dev_ptr))
    ptr = ctypes.cast(dev_ptr, ctypes.c_void_p)
    return ctypes.cast(ptr.value + shift, ctypes.POINTER(ctypes.c_ubyte))

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

    def all_data(self):
        return self.__data

    def dots(self, other):
        m = self.nvec()
        v = numpy.ndarray((m,), dtype = self.data_type())
        vdim = self.dimension()
        n = ctypes.c_int(vdim)
        inc = ctypes.c_int(1)
        vsize = self.__dsize * vdim
        data_u = self.all_data()
        data_v = other.all_data()
        iu = self.first()
        iv = other.first()
        for i in range(m):
            ptr_u = shifted_ptr(data_u, (iu + i)*vsize)
            ptr_v = shifted_ptr(data_v, (iv + i)*vsize)
            if self.is_complex():
                s = self.__complex()
                self.__cublas.dot \
                    (self.__cublas.handle, n, ptr_v, inc, ptr_u, inc, s)
                v[i] = s[0] + 1j * s[1]
            else:
                s = self.__float()
                self.__cublas.dot \
                    (self.__cublas.handle, n, ptr_v, inc, ptr_u, inc, ctypes.byref(s))
                v[i] = s.value
        return v

