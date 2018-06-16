# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, STFC
"""
import numpy
import scipy
from raleigh.solver import Problem, Solver, conjugate
from raleigh.vectors import Vectors
#from raleigh.ndarray.vectors import Vectors
#from raleigh.ndarray.numpy_vectors import Vectors

try:
    from raleigh.ndarray.mkl import mkl, Cblas
    HAVE_MKL = True
except:
    HAVE_MKL = False

class Operator:
    def __init__(self, a):
        self.a = a
#        self.type = type(a[0,0])
    def apply(self, x, y, transp = False):
        if transp:
            x.apply(self.a.T, y)
        else:
            x.apply(self.a, y)
#    def apply(self, x, y, transp = False):
#        u = x.data()
#        type_x = type(u[0,0])
#        mixed_types = type_x is not self.type
#        if mixed_types:
#            u = u.astype(self.type)
#        if transp:
#            v = numpy.dot(u, self.a)
#        else:
#            v = numpy.dot(u, self.a.T)
#        if mixed_types:
#            v = v.astype(type_x)
#        y.data()[:,:] = v

class OperatorSVD:
    def __init__(self, a):
        self.a = a
#        self.type = type(a[0,0])
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
#    def apply(self, x, y, transp = False):
#        u = x.data()
#        type_x = type(u[0,0])
#        mixed_types = type_x is not self.type
#        if mixed_types:
#            u = u.astype(self.type)
#        if transp:
#            w = numpy.dot(u, self.a)
#            v = numpy.dot(w, self.a.T)
#        else:
#            w = numpy.dot(u, self.a.T)
#            v = numpy.dot(w, self.a)
#        if mixed_types:
#            v = v.astype(type_x)
#        y.data()[:,:] = v
##        if mixed_types:
##            z = numpy.dot(u.astype(self.type), conjugate(self.a))
##            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
##        else:
##            z = numpy.dot(u, conjugate(self.a))
##            y.data()[:,:] = numpy.dot(z, self.a)

def compute_right(a, transp, opt, nsv, vtc = None):
    if transp:
        n, m = a.shape
    else:
        m, n = a.shape
    if vtc is None:
        v = Vectors(n)
    else:
        v = Vectors(vtc, with_mkl = False)
    operator = OperatorSVD(a)
    problem = Problem(v, lambda x, y: operator.apply(x, y, transp))
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

def partial_svd_old(a, opt, nsv = -1, uc = None, vtc = None):
    m, n = a.shape
    if m >= n:
        vt = compute_right(a, False, opt, nsv, vtc)
        sigma, u, vt = compute_left(a, vt)
        return sigma, u, vt
    else:
        print('transposing...')
        b = conjugate(a)
        if uc is not None:
            vtc = conjugate(uc)
        else:
            vtc = None
#        u = compute_right(b, opt, nsv, vtc)
        u = compute_right(a, True, opt, nsv, vtc)
        sigma, vt, u = compute_left(b, u)
        return sigma, conjugate(u), conjugate(vt)

def partial_svd(a, opt, nsv = -1, cstr = None):
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