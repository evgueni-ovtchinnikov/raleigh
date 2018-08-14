# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:06:19 2018

@author: wps46139
"""

raleigh_path = '../../..'

import numpy
import sys
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.cuda.cublas_algebra import Vectors, Matrix

class Operator:
    def __init__(self, array):
        self.matrix = Matrix(array)
    def apply(self, x, y, transp = False):
        self.matrix.apply(x, y, transp)

class OperatorSVD:
    def __init__(self, array):
        self.matrix = Matrix(array)
    def apply(self, x, y, transp = False):
        m, n = self.matrix.shape()
        k = x.nvec()
        if transp:
            z = Vectors(n, k, x.data_type())
            self.matrix.apply(x, z, transp = True)
            self.matrix.apply(z, y)
        else:
            z = Vectors(m, k, x.data_type())
            self.matrix.apply(x, z)
            self.matrix.apply(z, y, transp = True)

m = 3
n = 2

u = Vectors(numpy.ones((1, n)))
v = Vectors(numpy.ones((1, m)))

a = Operator(numpy.ones((m, n)))

#a.apply(u, v)
#
#print(v.data())
#
#a.apply(v, u, transp = True)
#
#print(u.data())

b = OperatorSVD(numpy.ones((m, n)))

x = Vectors(u)
y = Vectors(v)

b.apply(u, x)
print(x.data())

b.apply(v, y, transp = True)
print(y.data())
