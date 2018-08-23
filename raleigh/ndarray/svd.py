# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numpy
import scipy
import sys
import time

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Problem, Solver

class PSVDErrorEstimator:
    def __init__(self):
        self.reset()
    def reset(self):
        self.solver = None
        self.err = None
        self.op = None
        self.m = 0
        self.n = 0
        self.ncon = 0
    def set_up(self, op, solver, eigenvectors):
        self.op = op
        self.solver = solver
        self.eigenvectors = eigenvectors
        m, n = op.shape()
        if self.m != m or self.n != n:
            self.m = m
            self.n = n
            self.err = numpy.ones((m,))
        else:
            print('%d converged eigenvectors found' % self.ncon)
    def update(self):
        # TODO: update error
        self.ncon = self.eigenvectors.nvec()
        return self.err

def partial_svd(a, opt, nsv = -1, cstr = None, one_side = False, arch = 'cpu'):

    if arch[:3] == 'gpu':
        try:
            import raleigh.cuda.cuda as cuda
            from raleigh.cuda.cublas_algebra import Vectors, Matrix
            op = Matrix(a)
            gpu = True
        except:
            if len(arch) > 3 and arch[3] == '!':
                raise RuntimeError('cannot use GPU')
            gpu = False
    else:
        gpu = False
    if not gpu:
        from raleigh.algebra import Vectors, Matrix
        op = Matrix(a)

    class OperatorSVD:
        def __init__(self, op, gpu):
            self.op = op
            self.time = 0
        def apply(self, x, y, transp = False):
            m, n = self.op.shape()
            k = x.nvec()
            start = time.time()
            if transp:
                z = Vectors(n, k, x.data_type())
                self.op.apply(x, z, transp = True)
                self.op.apply(z, y)
            else:
                z = Vectors(m, k, x.data_type())
                self.op.apply(x, z)
                self.op.apply(z, y, transp = True)
            if gpu:
                cuda.synchronize()
            stop = time.time()
            self.time += stop - start

    m, n = a.shape
    transp = m < n
    if transp:
        n, m = m, n
    opSVD = OperatorSVD(op, gpu)
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
    try:
        opt.stopping_criteria.err_est.set_up(op, solver, v)
        print('partial SVD error estimation set up')
    except:
        print('partial SVD error estimation not requested')
        pass
    solver.solve(v, opt, which = (0, nsv))
    print('operator application time: %.2e' % opSVD.time)

    if one_side:
        sigma = -numpy.sort(-numpy.sqrt(solver.eigenvalues))
        # TODO: sort eigenvectors accordingly
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