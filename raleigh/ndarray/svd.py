# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numpy
import numpy.linalg as nla
import scipy
import sys
import time

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Problem, Solver

class PSVDErrorCalculator:
    def __init__(self, a):
#        self.reset()
#    def reset(self):
        self.solver = None
        self.err = None
        self.op = None
        m, n = a.shape
        self.m = m
        self.n = n
        self.ncon = 0
        self.norms = nla.norm(a, axis = 1)
        self.err = self.norms
    def set_up(self, op, solver, eigenvectors):
        self.op = op
        self.solver = solver
        self.eigenvectors = eigenvectors
    def update_errors(self):
        # TODO: update error
        ncon = self.eigenvectors.nvec()
        new = ncon - self.ncon
        if new > 0:
#            print('%d singular pairs converged' % new)
            x = self.eigenvectors
            sel = x.selected()
            x.select(new, self.ncon)
            m = self.m
            n = self.n
            if m < n:
                y = x.clone()
                lmd = self.solver.eigenvalues[self.ncon :]
                y.scale(lmd, multiply = True)
                q = x.dots(y, transp = True)
            else:
                y = x.new_vectors(new, m)
                self.op.apply(x, y)
                q = y.dots(y, transp = True)
            s = self.err*self.err - q
            s[s < 0] = 0
            self.err = numpy.sqrt(s)
            self.eigenvectors.select(sel[1], sel[0])
            self.ncon = ncon
        return self.err

def conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a

def partial_svd(a, opt, nsv = -1, isv = None, arch = 'cpu'):

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
#        from raleigh.algebra import Vectors, Matrix
        from raleigh.ndarray.numpy_algebra import Vectors, Matrix
        op = Matrix(a)

    class OperatorSVD:
        def __init__(self, op, gpu, transp = False):
            self.op = op
            self.gpu = gpu
            self.transp = transp
            self.time = 0
            m, n = self.op.shape()
            if transp:
                self.w = Vectors(n)
            else:
                self.w = Vectors(m)
        def apply(self, x, y):
            m, n = self.op.shape()
            k = x.nvec()
            start = time.time()
            if self.transp:
                if self.w.nvec() < k:
                    self.w = Vectors(n, k, x.data_type())
                z = self.w
                z.select(k)
                self.op.apply(x, z, transp = True)
                self.op.apply(z, y)
            else:
                if self.w.nvec() < k:
                    self.w = Vectors(m, k, x.data_type())
                z = self.w
                z.select(k)
                self.op.apply(x, z)
                self.op.apply(z, y, transp = True)
            if self.gpu:
                cuda.synchronize()
            stop = time.time()
            self.time += stop - start

    m, n = a.shape
    dt = a.dtype.type

    if isv is not None:
        k, l = isv.shape
        if k != n:
            msg = 'initial singular vectors must have dimension %d, not %d'
            raise ValueError(msg % (n, k))
        isv = Vectors(isv.T)

    transp = m < n
    if transp:
        n, m = m, n
        if isv is not None:
            tmp = Vectors(n, l, data_type = dt)
            op.apply(isv, tmp)
            isv = tmp

    opSVD = OperatorSVD(op, gpu, transp)
    v = Vectors(n, data_type = dt)
    problem = Problem(v, lambda x, y: opSVD.apply(x, y))
    solver = Solver(problem)

    try:
        opt.stopping_criteria.err_calc.set_up(op, solver, v)
        if opt.verbosity > 0:
            print('partial SVD error calculation set up')
    except:
        if opt.verbosity > 0:
            print('partial SVD error calculation not requested')
        pass

    solver.solve(v, opt, which = (0, nsv), init = (None, isv))
    if opt.verbosity > 0:
        print('operator application time: %.2e' % opSVD.time)

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
        return sigma, v.data().T, conj(u.data())
    else:
        return sigma, u.data().T, conj(v.data())