# -*- coding: utf-8 -*-
"""
Base class for ndarray implementation of Vectors and Matrix type

Created on Thu Jun 14 11:48:14 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numbers
import numpy

def fill_ndarray_with_orthogonal_vectors(a):
    m, n = a.shape
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
        return
    while j <= n:
        a[:k, i : j] = a[:k, :i]
        i, j = j, 2*j
    j = i//2
    a[k : m,   : j] = a[:(m - k), : j]
    a[k : m, j : i] = -a[:(m - k), j : i]

class NDArrayVectors:
    def __init__(self, arg, nvec = 0, data_type = None):
        if isinstance(arg, NDArrayVectors):
            #print('in copy constructor')
            i, n = arg.selected()
            self.__data = arg.__data[i : i + n, :].copy()
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
        dt = self.__data.dtype.type
        dk = self.__data.dtype.kind
        self.__dtype = dt
        self.__is_complex = (dk == 'c')
#        self.__is_complex = (dt == numpy.complex64 or dt == numpy.complex128)
        m, n = self.__data.shape
        self.__selected = (0, m)
        self.__vdim = n
        self.__nvec = m

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
    def fill_orthogonal(self):
        iv, nv = self.__selected
        k, n = self.__data.shape
        if n < nv:
            raise ValueError('fill_orthogonal: too many vectors in the array')
        fill_ndarray_with_orthogonal_vectors(self.__data[iv : iv + nv, :])
    def all_data(self):
        return self.__data
#    def data(self, i = None, transp = False):
    def data(self, i = None):
        f, n = self.__selected
        if i is None:
            return self.__data[f : f + n, :]
        else:
#            if transp:
#                return self.__data[f : f + n, i]
            return self.__data[f + i, :]
#    def data_slice(self, i):
#        f, n = self.__selected
#        return self.__data[f : f + n, i]
    def append(self, other):
        self.__data = numpy.concatenate((self.__data, other.data()))
        self.__nvec += other.nvec()
        self.select_all()

class NDArrayMatrix:
    def __init__(self, data):
        self.__data = data
        self.__shape = data.shape
        self.__dtype = data.dtype.type
    def data(self):
        return self.__data
    def shape(self):
        return self.__shape
    def data_type(self):
        return self.__dtype
    def is_complex(self):
        return (self.__data.dtype.kind == 'c')
