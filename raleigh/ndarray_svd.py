# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, UKRI
"""
import numpy
import scipy
from raleigh.solver import *
from raleigh.ndarray_vectors import NDArrayVectors

class Operator:
    def __init__(self, a):
        self.a = a
        self.type = type(a[0,0])
    def apply(self, x, y):
        u = x.data()
        type_x = type(u[0,0])
        if type_x is not self.type:
            mixed_types = True
            u = u.astype(self.type)
        else:
            mixed_types = False
        z = numpy.dot(u, self.a.T)
        if mixed_types:
            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
        else:
            y.data()[:,:] = numpy.dot(z, self.a)

def compute_right(a, opt):
    m, n = a.shape
    v = NDArrayVectors(n)
    operator = Operator(a)
    problem = Problem(v, lambda x, y: operator.apply(x, y))
    solver = Solver(problem)
    solver.solve(v, opt, which = (0, -1))
    return conjugate(v.data())

def compute_left(a, v):
    u = numpy.dot(a, v)
    vv = numpy.dot(conjugate(v), v)
    uu = -numpy.dot(conjugate(u), u)
    lmd, x = scipy.linalg.eigh(uu, vv, turbo = False)
    u = numpy.dot(u, x)
    v = numpy.dot(v, x)
    sigma = numpy.linalg.norm(u, axis = 0)
    u /= sigma
    return sigma, u

def ndarray_svd(a, opt):
    v = compute_right(a, opt)
    sigma, u = compute_left(a, v)
    return sigma, u, v
