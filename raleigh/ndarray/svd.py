# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, STFC
"""
import numpy
import scipy
from raleigh.solver import Problem, Solver
from raleigh.vectors import Vectors

class Operator:
    def __init__(self, a):
        self.a = a
    def apply(self, x, y, transp = False):
        if transp:
            x.apply(self.a.T, y)
        else:
            x.apply(self.a, y)

class OperatorSVD:
    def __init__(self, a):
        self.a = a
    def apply(self, x, y, transp = False):
        m, n = self.a.shape
        k = x.nvec()
        if transp:
            z = Vectors(n, k, x.data_type())
            x.apply(self.a.T, z)
            z.apply(self.a, y)
        else:
            z = Vectors(m, k, x.data_type())
            x.apply(self.a, z)
            z.apply(self.a.T, y)

def partial_svd(a, opt, nsv = -1, cstr = None, one_side = False):
    m, n = a.shape
    transp = m < n
    if transp:
        n, m = m, n
    op = Operator(a)
    opSVD = OperatorSVD(a)
    dt = a.dtype.type
    if cstr is None:
        v = Vectors(n, data_type = dt)
    else:
        if transp:
            v = Vectors(cstr[0].T, data_type = dt)
        else:
            v = Vectors(cstr[1], data_type = dt)
    problem = Problem(v, lambda x, y: opSVD.apply(x, y, transp))
    solver = Solver(problem)
    solver.solve(v, opt, which = (0, nsv))
    if one_side:
        sigma = numpy.sqrt(solver.eigenvalues)
        if transp:
            return sigma, v.data().T
        else:
            return sigma, v.data()
    nv = v.nvec()
    u = Vectors(m, nv, v.data_type())
    op.apply(v, u, transp)
    vv = v.dot(v)
    uu = -u.dot(u)
    lmd, x = scipy.linalg.eigh(uu, vv, turbo = False)
    w = v.new_vectors(nv)
    v.multiply(x, w)
    w.copy(v)
    w = u.new_vectors(nv)
    u.multiply(x, w)
    w.copy(u)
    sigma = numpy.sqrt(abs(u.dots(u)))
    u.scale(sigma)
    if transp:
        return sigma, v.data().T, u.data()
    else:
        return sigma, u.data().T, v.data()