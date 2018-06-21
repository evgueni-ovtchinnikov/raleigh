# -*- coding: utf-8 -*-
"""
Pure numpy implementation of ndarray-vectors algebra.

Created on Thu Jun 14 11:52:38 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numpy

from raleigh.ndarray.algebra_bc import NDArrayVectors, NDArrayMatrix

class Vectors(NDArrayVectors):
    def __init__(self, arg, nvec = 0, data_type = None):
        super(Vectors, self).__init__(arg, nvec, data_type)
    def new_vectors(self, nv = 0):
        return Vectors(self.dimension(), nv, self.data_type())
    def clone(self):
        return Vectors(self)

    # BLAS level 1
    def copy(self, other, ind = None):
        i, n = self.selected()
        j, m = other.selected()
        if ind is None:
            assert(m == n)
            other.data()[:,:] = self.data()
        else:
            other.all_data()[j : j + len(ind), :] = self.all_data()[ind, :]
    def scale(self, s):
        f, n = self.selected();
        for i in range(n):
            if s[i] != 0.0:
                self.data(i)[:] /= s[i]
    def dots(self, other):
        n = self.nvec()
        v = numpy.ndarray((n,), dtype = self.data_type())
        for i in range(n):
            if other.is_complex():
                s = numpy.dot(other.data(i).conj(), self.data(i))
            else:
                s = numpy.dot(other.data(i), self.data(i))
            v[i] = s
        return v

    # BLAS level 3
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
            numpy.dot(q.T, self.data(), out = output.data())
            return
        print('using non-optimized dot')
        output.data()[:,:] = numpy.dot(q.T, self.data())
    def apply(self, A, output, transp = False):
        a = A.data()
        if transp:
            is_complex = isinstance(a[0,0], complex)
            if is_complex:
                numpy.conj(self.data(), out = self.data())
            self.__apply(a.T, output)
            if is_complex:
                numpy.conj(output.data(), out = output.data())
                numpy.conj(self.data(), out = self.data())
        else:
            self.__apply(a, output)
    def __apply(self, q, output):
        if output.data().flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(self.data(), q.T, out = output.data())
        else:
            print('using non-optimized dot')
            output.data()[:,:] = numpy.dot(self.data(), q.T)
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

class Matrix(NDArrayMatrix):
    pass
