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
            raise ValueError('data type %s not supported' % repr(data_type))
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
            self.__data = arg
            self.__is_complex = isinstance(arg[0,0], complex)
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

    def dimension(self):
        return self.__vdim
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
