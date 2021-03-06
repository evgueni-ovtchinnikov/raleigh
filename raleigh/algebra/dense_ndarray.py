# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Base classes for ndarray implementation of Vectors and Matrix types.
"""

import numbers
import numpy


class NDArrayVectors(object):
    '''Base class for ndarray-based Vectors type.
    '''

    '''========== Methods required by RALEIGH core solver ==========
    '''

    def dimension(self):
        return self.__vdim

    def nvec(self):
        return self.__selected[1]

    def select(self, nv, first=0):
        assert nv <= self.__nvec and first >= 0
        self.__selected = (first, nv)

    def selected(self):
        return self.__selected

    def data_type(self):
        return self.__dtype

    def fill_random(self):
        iv, nv = self.__selected
        m, n = self.__data.shape
        self.__data[iv : iv + nv, :] = 2*numpy.random.rand(nv, n) - 1

    def append(self, other, axis=0):
        if axis == 0:
            self.__data = numpy.concatenate((self.data(), other.data()))
            self.__nvec = self.nvec() + other.nvec()
            self.select_all()
        else:
            self.__data = numpy.concatenate((self.__data, other.all_data()), \
                                             axis=1)
            self.__vdim += other.dimension()

    '''========== Other methods ====================================
    '''

    def __init__(self, arg, nvec=0, data_type=None, shallow=False):
        if isinstance(arg, NDArrayVectors):
            i, n = arg.selected()
            if shallow:
                self.__data = arg.__data[i : i + n, :]
            else:
                self.__data = arg.__data[i : i + n, :].copy()
        elif isinstance(arg, NDArrayMatrix):
            if shallow:
                self.__data = arg.data()
            else:
                self.__data = arg.data().copy()
            if not self.__data.flags['C_CONTIGUOUS']:
                raise ValueError('Vectors data must be C_CONTIGUOUS')
        elif isinstance(arg, numpy.ndarray):
            self.__data = arg
        elif isinstance(arg, numbers.Number):
            if data_type is None: # use default data type
                self.__data = numpy.zeros((nvec, arg))
            else:
                self.__data = numpy.zeros((nvec, arg), dtype=data_type)
        else:
            raise ValueError \
                ('wrong argument %s in constructor' % repr(type(arg)))
        dt = self.__data.dtype.type
        dk = self.__data.dtype.kind
        self.__dtype = dt
        self.__is_complex = (dk == 'c')
        m, n = self.__data.shape
        self.__selected = (0, m)
        self.__vdim = n
        self.__nvec = m

    def select_all(self):
        self.select(self.__nvec)

    def is_complex(self):
        return self.__is_complex

    def zero(self):
        f, n = self.__selected;
        self.__data[f : f + n, :] = 0.0

    def fill(self, array_or_value):
        f, n = self.__selected;
        self.__data[f : f + n, :] = array_or_value

    def fill_orthogonal(self):
        iv, nv = self.__selected
        k, n = self.__data.shape
        if n < nv:
            raise ValueError('fill_orthogonal: too many vectors in the array')
        _fill_ndarray_with_orthogonal_vectors(self.__data[iv : iv + nv, :])

    def all_data(self):
        return self.__data

    def data(self, i=None):
        f, n = self.__selected
        if i is None:
            return self.__data[f : f + n, :]
        else:
            return self.__data[f + i, :]


class NDArrayMatrix:

    def __init__(self, arg):
        if isinstance(arg, NDArrayVectors):
            data = arg.data()
        elif isinstance(arg, numpy.ndarray):
            data = arg
        else:
            raise ValueError \
                ('wrong argument %s in Matrix constructor' % repr(type(arg)))
        self.__data = data
        self.__shape = data.shape
        self.__dtype = data.dtype.type
        if data.flags['C_CONTIGUOUS']:
            self.__order = 'C_CONTIGUOUS'
        elif data.flags['F_CONTIGUOUS']:
            self.__order = 'F_CONTIGUOUS'
        else:
            msg = 'Matrix data must be either C- or F-contiguous'
            raise ValueError(msg)

    def data(self):
        return self.__data

    def shape(self):
        return self.__shape

    def data_type(self):
        return self.__dtype

    def is_complex(self):
        return (self.__data.dtype.kind == 'c')

    def order(self):
        return self.__order


def _fill_ndarray_with_orthogonal_vectors(a):
    a.fill(0.0)
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
    a[k : m,   : j] = a[: (m - k), : j]
    a[k : m, j : i] = -a[: (m - k), j : i]
