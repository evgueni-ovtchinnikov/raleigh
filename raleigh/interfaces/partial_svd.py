# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Partial SVD of a matrix represented by a 2D ndarray.

   For advanced users only.
"""

import math
import numpy
import numpy.linalg as nla
import scipy
import time

from ..core.solver import Problem, Solver, Options


class PartialSVD:

    def __init__(self, matrix, shift=False):
        op = matrix.as_operator()
        m, n = matrix.shape()
        transp = m < n
        if transp:
            n, m = m, n
        v = op.new_vectors(n)
        dt = v.data_type()
        opSVD = _OperatorSVD(matrix, v, transp, shift)
        self.__op = op
        self.__transp = transp
        self.__shape = (m, n)
        self.__shift = shift
        self.__v = v
        self.__dtype = dt
        self.__opsvd = opSVD
        self.sigma = None
        self.__left = None
        self.__right = None
        self.__mean = None
        self.__left_v = None
        self.__right_v = None
        self.__mean_v = None
        self.iterations = -1

    def op_svd(self):
        return self.__opsvd

    def vectors(self):
        return self.__v

    def compute(self, matrix, opt=Options(), nsv=(-1, -1), refine=True):
    
        op = self.__op
        m, n = self.__shape
        transp = self.__transp
        v = self.__v
        dt = self.__dtype
        opSVD = self.__opsvd
        shift = self.__shift

        problem = Problem(v, opSVD)
        solver = Solver(problem)

        status = solver.solve(v, options=opt, which=nsv)
        if status < 0:
            self.__mean_v = None
            self.__left_v = None
            self.__right_v = None
            return
        if opt.verbosity > 0:
            print('operator application time: %.2e' % opSVD.time)

        nv = v.nvec()
        u = v.new_vectors(nv, m)
        if nv > 0:
            op.apply(v, u, transp)
            if shift:
                m, n = op.shape()
                dt = op.data_type()
                ones = numpy.ones((1, m), dtype=dt)
                e = v.new_vectors(1, m)
                e.fill(ones)
                w = v.new_vectors(1, n)
                op.apply(e, w, transp=True)
                w.scale(m*ones[0,:1])
                if not transp:
                    s = v.dot(w)
                    u.add(e, -1, s)
                else:
                    s = v.dot(e)
                    u.add(w, -1, s)
            if refine:
                sigma, q = u.svd()
                w = v.new_vectors(nv)
                v.multiply(q, w)
                w.copy(v)
            else:
                sigma = numpy.sqrt(abs(u.dots(u)))
                u.scale(sigma)
                w = u.new_vectors(nv)
                ind = numpy.argsort(-sigma)
                sigma = sigma[ind]
                u.copy(w, ind)
                w.copy(u)
                w = v.new_vectors(nv)
                v.copy(w, ind)
                w.copy(v)
        else:
            sigma = numpy.ndarray((0,), dtype=v.data_type())
        self.sigma = sigma
        self.__mean_v = opSVD.mean_v()
        self.iterations = solver.iteration
        if transp:
            self.__left_v = v
            self.__right_v = u
        else:
            self.__left_v = u
            self.__right_v = v

    def mean(self):
        if self.__mean is None:
            if self.__mean_v is not None:
                self.__mean = self.__mean_v.data()
        return self.__mean

    def left(self):
        if self.__left is None:
            if self.__left_v is not None:
                self.__left = self.__left_v.data().T
        return self.__left

    def right(self):
        if self.__right is None:
            if self.__right_v is not None:
                self.__right = self.__right_v.data().T
        return self.__right

    def mean_v(self):
        return self.__mean_v

    def left_v(self):
        return self.__left_v

    def right_v(self):
        return self.__right_v

    def _finalize_svd(v, Av):
        '''Given right singular vectors v of A and Av, compute left singular vectors
           u and singular values sigma, and adjust v so that A v = u sigma.
           Try to do it fast if possible, otherwise do SVD of Av.
        '''
        nsv = v.nvec()
        Gram = Av.dot(Av)
        diag = numpy.diag(Gram)
        if numpy.amin(diag) == 0.0:
            icond = 0.0
        else:
            Diag = numpy.diag(diag)
            lmd, _ = sla.eigh(Gram, Diag)
            icond = lmd[0]/lmd[-1]
        eps = 100*numpy.finfo(diag.dtype).eps
        if icond < eps: # Av is too ill-conditioned, use SVD of Av
            sigma, q = Av.svd()
            u = Av
            w = v.new_vectors(nsv)
            v.multiply(q, w)
            w.copy(v)
            return u, sigma, v
        # Av not too bad, try faster route
        w = Av.new_vectors(nsv)
        U = _conj(nla.cholesky(Gram).T) # Gram = L L.H = U.H U
        Ui = sla.inv(U)
        Av.multiply(Ui, w) # A v = w U
        p, sigma, qt = sla.svd(U) # A v = w p sigma qt = u sigma q.H
        q = _conj(qt.T) # q = qt.H
        u = Av # Av no longer needed, let us recycle it as u
        w.multiply(p, u)
        Gram = u.dot(u)
        no_max = nla.norm(Gram - numpy.eye(u.nvec()))
        maxit = 2
        it = 0
        while no_max > eps and it < maxit:
            U = _conj(nla.cholesky(Gram).T) # Gram = L L.H = U.H U
            Ui = sla.inv(U)
            u.multiply(Ui, w) # A v = u sigma q.H = w U sigma q.H
            p, sigma, qh = sla.svd(U*sigma)
            # A v = w p sigma gh q.H = w p sigma (q qh.H).H = u sigma q.H
            q = numpy.dot(q, _conj(qt.T))
            w.multiply(p, u)
            Gram = u.dot(u)
            no_max = nla.norm(Gram - numpy.eye(u.nvec()))
            it += 1
        w = v.new_vectors(v.nvec())
        v.multiply(q, w)
        w.copy(v)
        return u, sigma, v


class _OperatorSVD:
    def __init__(self, matrix, v, transp=False, shift=False):
        self.op = matrix.as_operator()
        self.gpu = matrix.gpu()
        self.transp = transp
        self.shift = shift
        self.time = 0
        m, n = self.op.shape()
        if transp:
            self.w = v.new_vectors(0, n)
        else:
            self.w = v.new_vectors(0, m)
        if shift:
            dt = self.op.data_type()
            ones = numpy.ones((1, m), dtype=dt)
            self.ones = v.new_vectors(1, m)
            self.ones.fill(ones)
            self.aves = v.new_vectors(1, n)
            self.op.apply(self.ones, self.aves, transp=True)
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
            self.op.apply(x, z, transp=True)
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
            self.op.apply(z, y, transp=True)
        if self.gpu is not None:
            self.gpu.synchronize()
        stop = time.time()
        self.time += stop - start
    def mean(self):
        if self.shift:
            return self.aves.data()
        else:
            return None
    def mean_v(self):
        if self.shift:
            return self.aves
        else:
            return None


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a
