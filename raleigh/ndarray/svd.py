# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, STFC
"""

import numpy
import scipy
import sys

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Problem, Solver

def partial_svd(a, opt, nsv = -1, cstr = None, one_side = False, try_gpu = False):

    if try_gpu:
        try:
            from raleigh.cuda.cublas_algebra import Vectors, Matrix
            op = Matrix(a)
            gpu = True
        except:
            gpu = False
    else:
        gpu = False
    if not gpu:
        from raleigh.algebra import Vectors, Matrix
        op = Matrix(a)

#    class Operator:
#        def __init__(self, array):
#            self.matrix = Matrix(array)
#        def apply(self, x, y, transp = False):
#            self.matrix.apply(x, y, transp)    
    class OperatorSVD:
#        def __init__(self, array):
#            self.matrix = Matrix(array)
        def __init__(self, op):
            self.op = op
        def apply(self, x, y, transp = False):
#            m, n = self.matrix.shape()
            m, n = self.op.shape()
            k = x.nvec()
            if transp:
                z = Vectors(n, k, x.data_type())
                self.op.apply(x, z, transp = True)
                self.op.apply(z, y)
#                self.matrix.apply(x, z, transp = True)
#                self.matrix.apply(z, y)
            else:
                z = Vectors(m, k, x.data_type())
                self.op.apply(x, z)
                self.op.apply(z, y, transp = True)
#                self.matrix.apply(x, z)
#                self.matrix.apply(z, y, transp = True)

    m, n = a.shape
    transp = m < n
    if transp:
        n, m = m, n
#    opSVD = OperatorSVD(a)
    opSVD = OperatorSVD(op)
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