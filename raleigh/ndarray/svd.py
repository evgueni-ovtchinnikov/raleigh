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

try:
    from raleigh.ndarray.mkl import mkl, Cblas
    HAVE_MKL = True
except:
    HAVE_MKL = False

class Operator:
    def __init__(self, a):
        self.a = a
        self.type = type(a[0,0])
    def apply(self, x, y):
        u = x.data()
        type_x = type(u[0,0])
        mixed_types = type_x is not self.type
        if mixed_types:
            z = numpy.dot(u.astype(self.type), conjugate(self.a))
            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
        else:
            z = numpy.dot(u, conjugate(self.a))
            y.data()[:,:] = numpy.dot(z, self.a)

def compute_right(a, opt, nsv, vtc = None):
    m, n = a.shape
    if vtc is None:
        v = Vectors(n)
    else:
        v = Vectors(vtc, with_mkl = False)
    operator = Operator(a)
    problem = Problem(v, lambda x, y: operator.apply(x, y))
    solver = Solver(problem)
    solver.solve(v, opt, which = (0, nsv))
    vt = v.data()
    return vt

def compute_left(a, vt):
    v = conjugate(vt)
    u = numpy.dot(a, v)
    vv = numpy.dot(conjugate(v), v)
    uu = -numpy.dot(conjugate(u), u)
    lmd, x = scipy.linalg.eigh(uu, vv, turbo = False)
    u = numpy.dot(u, x)
    v = numpy.dot(v, x)
    sigma = numpy.linalg.norm(u, axis = 0)
    u /= sigma
    return sigma, u, conjugate(v)

def partial_svd(a, opt, nsv = -1, uc = None, vtc = None):
    m, n = a.shape
    if m >= n:
        vt = compute_right(a, opt, nsv, vtc)
        sigma, u, vt = compute_left(a, vt)
        return sigma, u, vt
    else:
        b = conjugate(a)
        if uc is not None:
            vtc = conjugate(uc)
        else:
            vtc = None
        u = compute_right(b, opt, nsv, vtc)
        sigma, vt, u = compute_left(b, u)
        return sigma, conjugate(u), conjugate(vt)
