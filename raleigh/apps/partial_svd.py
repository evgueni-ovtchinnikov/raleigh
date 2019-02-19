# -*- coding: utf-8 -*-
"""Partial SVD of an ndarray via RALEIGH

Created on Tue Feb 19 13:58:54 2019

@author: Evgueni Ovtchinnikov, UKRI
"""

import numpy
import scipy
import time

from ..solver import Problem, Solver


class PartialSVD:
    def __init__(self):
        self.sigma = None
        self.u = None
        self.v = None
        self.mean = None
        self.iterations = -1
    def compute(self, a, opt, nsv = (-1, -1), isv = (None, None), \
                shift = False, refine = False, arch = 'cpu'):
    
        if arch[:3] == 'gpu':
            try:
                from ..cuda import cuda
                from ..cuda.cublas_algebra import Vectors, Matrix
                op = Matrix(a)
                gpu = cuda
            except:
                if len(arch) > 3 and arch[3] == '!':
                    raise RuntimeError('cannot use GPU')
                gpu = None
        else:
            gpu = None
        if gpu is None:
            from ..algebra import Vectors, Matrix
            op = Matrix(a)
    
        m, n = a.shape
        dt = a.dtype.type
    
        isvec = ()
        for i in range(2):
            if isv[i] is not None:
                k, l = isv[i].shape
                if k != n:
                    msg = 'initial singular vectors must have dimension %d, not %d'
                    raise ValueError(msg % (n, k))
                isvec += (Vectors(isv[i].T),)
            else:
                isvec += (None,)
        isv = isvec
    
        transp = m < n
        if transp:
            n, m = m, n
            isvec = ()
            for i in range(2):
                if isv[i] is not None:
                    tmp = Vectors(n, l, data_type = dt)
                    op.apply(isv[i], tmp)
                    isvec += (tmp,)
                else:
                    isvec += (None,)
            isv = isvec
    
        v = Vectors(n, data_type = dt)
        opSVD = _OperatorSVD(op, v, gpu, transp, shift)
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
                ones = numpy.ones((1, m), dtype = dt)
                e = Vectors(m, 1, data_type = dt)
                e.fill(ones)
                w = Vectors(n, 1, data_type = dt)
                op.apply(e, w, transp = True)
                w.scale(m*ones[0,:1])
                if not transp:
                    s = v.dot(w)
                    u.add(e, -1, s)
                else:
                    s = v.dot(e)
                    u.add(w, -1, s)
            if refine:
                vv = v.dot(v)
                if nsv[0] == 0:
                    uu = -u.dot(u)
                else:
                    uu = u.dot(u)
                lmd, x = scipy.linalg.eigh(uu, vv) #, turbo = False)
                w = v.new_vectors(nv)
                v.multiply(x, w)
                w.copy(v)
                w = u.new_vectors(nv)
                u.multiply(x, w)
                w.copy(u)
            sigma = numpy.sqrt(abs(u.dots(u)))
            u.scale(sigma)
            if refine:
                u = u.data().T
                v = v.data().T
            else:
                ind = numpy.argsort(-sigma)
                sigma = sigma[ind]
                u = u.data().T[:, ind]
                v = v.data().T[:, ind]
        else:
            sigma = numpy.ndarray((0,), dtype = v.data_type())
            u = None
            v = None
        self.sigma = sigma
        self.mean = opSVD.mean()
        self.iterations = solver.iteration
        if transp:
            self.u = v
            self.v = u
            return sigma, v, _conj(u.T)
        else:
            self.u = u
            self.v = v
            return sigma, u, _conj(v.T)


class _OperatorSVD:
    def __init__(self, op, v, gpu, transp = False, shift = False):
        self.op = op
        self.gpu = gpu
        self.transp = transp
        self.shift = shift
        self.time = 0
        m, n = self.op.shape()
        if transp:
            self.w = v.new_vectors(0, n)
        else:
            self.w = v.new_vectors(0, m)
        if shift:
            dt = op.data_type()
            ones = numpy.ones((1, m), dtype = dt)
            self.ones = v.new_vectors(1, m)
            self.ones.fill(ones)
            self.aves = v.new_vectors(1, n)
            self.op.apply(self.ones, self.aves, transp = True)
            self.aves.scale(m*ones[0,:1])
    def apply(self, x, y):
        m, n = self.op.shape()
        k = x.nvec()
        start = time.time()
        if self.transp:
            if self.w.nvec() < k:
                self.w = x.new_vectors(k, n)
            z = self.w
            z.select(k)
            self.op.apply(x, z, transp = True)
            if self.shift:
                s = x.dot(self.ones)
                z.add(self.aves, -1, s)
            self.op.apply(z, y)
            if self.shift:
                s = z.dot(self.aves)
                y.add(self.ones, -1, s)
        else:
            if self.w.nvec() < k:
                self.w = x.new_vectors(k, m)
            z = self.w
            z.select(k)
            self.op.apply(x, z)
            if self.shift:
                s = z.dot(self.ones)
                z.add(self.ones, -1.0/m, s)
                # accurate orthogonalization needed!
                s = z.dot(self.ones)
                z.add(self.ones, -1.0/m, s)
            self.op.apply(z, y, transp = True)
        if self.gpu is not None:
            self.gpu.synchronize()
        stop = time.time()
        self.time += stop - start
    def mean(self):
        if self.shift:
            return self.aves
        else:
            return None


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a