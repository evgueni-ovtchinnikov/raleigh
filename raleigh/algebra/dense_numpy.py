# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)
# This software is distributed under a BSD licence, see ../../LICENSE.txt.
"""Pure numpy implementation of RALEIGH dense algebra.

Created on Thu Jun 14 11:52:38 2018
"""

import numpy

from .dense_ndarray import NDArrayVectors, NDArrayMatrix


class Vectors(NDArrayVectors):
    '''Pure numpy implementation of ndarray-based Vectors type.
    '''

    '''========== Methods required by RALEIGH core solver ==========
    '''

    def new_vectors(self, nv=0, dim=None):
        if dim is None:
            dim = self.dimension()
        return Vectors(dim, nv, self.data_type())

    def clone(self):
        return Vectors(self)

    def copy(self, other, ind=None):
        i, n = self.selected()
        j, m = other.selected()
        if ind is None:
            assert(m == n)
            other.data()[:,:] = self.data()
        else:
            other.all_data()[j : j + len(ind), :] = self.all_data()[ind, :]

    def scale(self, s, multiply=False):
        f, n = self.selected();
        if multiply:
            for i in range(n):
                self.data(i)[:] *= s[i]
        else:
            for i in range(n):
                if s[i] != 0.0:
                    self.data(i)[:] /= s[i]

    def dots(self, other, transp=False):
        if transp:
            u = self.data()
            v = other.data()
            n = self.dimension()
            w = numpy.ndarray((n,), dtype=self.data_type())
            if other.is_complex():
                for i in range(n):
                    w[i] = numpy.dot(v[:, i].conj(), u[:, i])
            else:
                for i in range(n):
                    w[i] = numpy.dot(v[:, i], u[:, i])
            return w
        else:
            n = self.nvec()
        v = numpy.ndarray((n,), dtype=self.data_type())
        for i in range(n):
            if other.is_complex():
                s = numpy.dot(other.data(i).conj(), self.data(i))
            else:
                s = numpy.dot(other.data(i), self.data(i))
            v[i] = s
        return v

    def dot(self, other):
        if other.is_complex():
            return numpy.dot(other.data().conj(), self.data().T)
        else:
            return numpy.dot(other.data(), self.data().T)

    def multiply(self, q, output):
        f, s = output.selected();
        m = q.shape[1]
        assert(s == m)
        if output.data().flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(q.T, self.data(), out=output.data())
            return
        print('using non-optimized dot')
        output.data()[:,:] = numpy.dot(q.T, self.data())

    def add(self, other, s, q = None):
        f, m = self.selected();
        if numpy.isscalar(s):
            if q is None:
                self.data()[:,:] += s*other.data()
            else:
                self.data()[:,:] += s*numpy.dot(q.T, other.data())
            return
        else:
            for i in range(m):
                self.data(i)[:] += s[i]*other.data(i)

    '''========== Other methods ====================================
    '''

    def __init__(self, arg, nvec=0, data_type=None, shallow=False):
        super(Vectors, self).__init__(arg, nvec, data_type, shallow)

    def apply(self, A, output, transp=False):
        a = A.data()
        if transp:
            is_complex = A.is_complex()
            if is_complex:
                numpy.conj(self.data(), out=self.data())
            self.__apply(a.T, output)
            if is_complex:
                numpy.conj(output.data(), out=output.data())
                numpy.conj(self.data(), out=self.data())
        else:
            self.__apply(a, output)
    def __apply(self, q, output):
        if output.data().flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(self.data(), q.T, out=output.data())
        else:
            #print('using non-optimized dot')
            output.data()[:,:] = numpy.dot(self.data(), q.T)


class Matrix(NDArrayMatrix):
    def apply(self, x, y, transp=False):
        if transp:
            is_complex = self.is_complex()
            if is_complex:
                numpy.conj(x.data(), out=x.data())
            self.__apply(x, y, transp)
            if is_complex:
                numpy.conj(x.data(), out=x.data())
                numpy.conj(y.data(), out=y.data())
        else:
            self.__apply(x, y)
    def __apply(self, x, y, transp=False):
        if transp:
            a = self.data().T
        else:
            a = self.data()
        if y.data().flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(x.data(), a.T, out=y.data())
        else:
            print('using non-optimized dot')
            y.data()[:,:] = numpy.dot(x.data(), a.T)
