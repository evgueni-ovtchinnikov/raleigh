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
        self.dt = a.dtype.type
        self.ncon = 0
        self.norms = nla.norm(a, axis = 1)
#        self.err = self.norms.copy()
    def set_up(self, op, solver, eigenvectors, shift = False):
        self.op = op
        self.solver = solver
        self.eigenvectors = eigenvectors
        if shift:
            self.ones = eigenvectors.new_vectors(1, self.n)
            ones = numpy.ones((1, self.n), dtype = eigenvectors.data_type())
            self.ones.fill(ones)
            self.aves = eigenvectors.new_vectors(1, self.m)
            self.op.apply(self.ones, self.aves)
            aves = numpy.zeros((self.m,), dtype = eigenvectors.data_type())
            aves[:] = self.aves.data()
            s = self.norms*self.norms - aves*aves/self.n
            self.norms = numpy.sqrt(abs(s))
            self.aves.scale(self.n*ones[0,:1])
        self.err = self.norms.copy()
    def update_errors(self):
        ncon = self.eigenvectors.nvec()
        new = ncon - self.ncon
        if new > 0:
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
                if self.shift:
                    s = x.dot(self.ones)
                    y.add(self.aves, -1, s)
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

def partial_svd(a, opt, nsv = (-1, -1), isv = (None, None), shift = False, \
                arch = 'cpu'):

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
#        from raleigh.ndarray.numpy_algebra import Vectors, Matrix
        op = Matrix(a)

    class OperatorSVD:
        def __init__(self, op, gpu, transp = False, shift = False):
            self.op = op
            self.gpu = gpu
            self.transp = transp
            self.shift = shift
            self.time = 0
            m, n = self.op.shape()
            if transp:
                self.w = Vectors(n)
            else:
                self.w = Vectors(m)
            if shift:
                dt = op.data_type()
                ones = numpy.ones((1, n), dtype = dt)
                self.ones = Vectors(n, 1, data_type = dt)
                self.ones.fill(ones)
                self.aves = Vectors(m, 1, data_type = dt)
                self.op.apply(self.ones, self.aves)
                self.aves.scale(n*ones[0,:1])
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
                if self.shift:
                    s = x.dot(self.aves)*n
                    y.add(self.aves, -1, s)
            else:
                if self.w.nvec() < k:
                    self.w = Vectors(m, k, x.data_type())
                z = self.w
                z.select(k)
                self.op.apply(x, z)
                if self.shift:
                    s = x.dot(self.ones)
                    z.add(self.aves, -1, s)
                self.op.apply(z, y, transp = True)
                if self.shift:
                    s = z.dot(self.aves)
                    y.add(self.ones, -1, s)
            if self.gpu:
                cuda.synchronize()
            stop = time.time()
            self.time += stop - start

    m, n = a.shape
    dt = a.dtype.type

    for i in range(2):
        if isv[i] is not None:
            k, l = isv[i].shape
            if k != n:
                msg = 'initial singular vectors must have dimension %d, not %d'
                raise ValueError(msg % (n, k))
            isv[i] = Vectors(isv[i].T)

    transp = m < n
    if transp:
        n, m = m, n
        for i in range(2):
            if isv[i] is not None:
                tmp = Vectors(n, l, data_type = dt)
                op.apply(isv[i], tmp)
                isv[i] = tmp

    opSVD = OperatorSVD(op, gpu, transp, shift)
    v = Vectors(n, data_type = dt)
    problem = Problem(v, lambda x, y: opSVD.apply(x, y))
    solver = Solver(problem)

    try:
        opt.stopping_criteria.err_calc.set_up(op, solver, v, shift)
        if opt.verbosity > 0:
            print('partial SVD error calculation set up')
    except:
        if opt.verbosity > 0:
            print('partial SVD error calculation not requested')
        pass

    #solver.solve(v, opt, which = (0, nsv), init = (None, isv))
    solver.solve(v, opt, which = nsv, init = isv)
    if opt.verbosity > 0:
        print('operator application time: %.2e' % opSVD.time)

    nv = v.nvec()
    u = Vectors(m, nv, v.data_type())
    if nv > 0:
        op.apply(v, u, transp)
        if shift:
            m, n = a.shape
            dt = op.data_type()
            ones = numpy.ones((1, n), dtype = dt)
            e = Vectors(n, 1, data_type = dt)
            e.fill(ones)
            w = Vectors(m, 1, data_type = dt)
            op.apply(e, w)
            w.scale(n*ones[0,:1])
            if transp:
                s = v.dot(w)
                u.add(e, -1, s)
            else:
                s = v.dot(e)
                u.add(w, -1, s)

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
    else:
        sigma = numpy.ndarray((0,), dtype = v.data_type())
    if transp:
        return sigma, v.data().T, conj(u.data())
    else:
        return sigma, u.data().T, conj(v.data())

def truncated_svd(a, opt, nsv = -1, isv = None, shift = False, arch = 'cpu'):
    return partial_svd(a, opt, (0, nsv), (None, isv), shift, arch)
