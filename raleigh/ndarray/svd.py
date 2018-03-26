# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, STFC
"""
import numpy
import scipy
from raleigh.solver import Problem, Solver, conjugate
from raleigh.ndarray.vectors import Vectors

class Operator:
    def __init__(self, a):
        self.a = a
        self.type = type(a[0,0])
    def apply(self, x, y):
        u = x.data()
        type_x = type(u[0,0])
        mixed_types = type_x is not self.type
        if mixed_types:
            z = numpy.dot(u.astype(self.type), self.a.T)
            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
        else:
            z = numpy.dot(u, self.a.T)
            y.data()[:,:] = numpy.dot(z, self.a)

def compute_right(a, opt):
    m, n = a.shape
    v = Vectors(n)
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

def partial_svd(a, opt):
    v = compute_right(a, opt)
    sigma, u = compute_left(a, v)
    return sigma, u, v
