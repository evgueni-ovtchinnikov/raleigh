# -*- coding: utf-8 -*-
"""Lower Rank Approximation of a dense matrix

Created on Tue Feb 19 14:07:09 2019

@author: Evgueni Ovtchinnikov, UKRI
"""

import copy
import numpy
import numpy.linalg as nla
import time

from .partial_svd import PartialSVD
from ..solver import Options

def pca(a, opt = Options(), npc = -1, tol = 0, th = 0, msv = 0, ipc = None, \
        arch = 'cpu'):
    lra = LowerRankApproximation()
    return lra.compute(a, opt = opt, nsv = npc, tol = tol, th = th, \
        msv = msv, isv = ipc, shift = True, arch = arch)


class LowerRankApproximation:
    def __init__(self):
        self.sigma = None
        self.u = None
        self.v = None
        self.iterations = -1
    def compute(self, a, opt, nsv = -1, tol = 0, th = 0, msv = 0, rtol = 1e-3, \
                  isv = None, shift = False, refine = False, \
                  arch = 'cpu'):
        m, n = a.shape
        opt = copy.deepcopy(opt)
        if opt.block_size < 1:
            opt.block_size = 128
        if opt.max_iter < 0:
            opt.max_iter = max(100, min(m, n))
        if opt.convergence_criteria is None:
            opt.convergence_criteria = _DefaultConvergenceCriteria(rtol)
        if opt.stopping_criteria is None and nsv < 0:
            opt.stopping_criteria = _DefaultStoppingCriteria(a, tol, th, msv)
        psvd = PartialSVD()
        psvd.compute(a, opt, (0, nsv), (None, isv), shift, refine, arch)
        self.sigma = psvd.sigma
        self.u = psvd.u
        self.v = psvd.v
        self.mean = psvd.mean
        self.iterations = psvd.iterations
        return self.sigma, self.u, self.v.T


class _PSVDErrorCalculator:
    def __init__(self, a):
        self.solver = None
        self.err = None
        self.op = None
        m, n = a.shape
        self.m = m
        self.n = n
        self.dt = a.dtype.type
        self.shift = False
        self.ncon = 0
        self.norms = nla.norm(a, axis = 1).reshape((m, 1))
        self.err = self.norms.copy()
    def set_up(self, op, solver, eigenvectors, shift = False):
        self.op = op
        self.solver = solver
        self.eigenvectors = eigenvectors
        self.shift = shift
        if shift:
            self.ones = eigenvectors.new_vectors(1, self.m)
            self.ones.fill(1.0)
            self.aves = eigenvectors.new_vectors(1, self.n)
            self.op.apply(self.ones, self.aves, transp = True)
            self.aves.scale(self.m*numpy.ones((1,)))
            s = self.aves.dots(self.aves)
            vb = eigenvectors.new_vectors(1, self.m)
            self.op.apply(self.aves, vb)
            b = vb.data().reshape((self.m, 1))
            t = (self.norms*self.norms).reshape((self.m, 1))
            x = t - 2*b + s*numpy.ones((self.m, 1))
            self.err = numpy.sqrt(abs(x))
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
                z = x.new_vectors(new, n)
                self.op.apply(x, z, transp = True)
                if self.shift:
                    s = x.dot(self.ones)
                    z.add(self.aves, -1, s)
                y = x.new_vectors(new, m)
                self.op.apply(z, y)
                if self.shift:
                    s = z.dot(self.aves)
                    y.add(self.ones, -1, s)
                q = x.dots(y, transp = True)
            else:
                y = x.new_vectors(new, m)
                self.op.apply(x, y)
                if self.shift:
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                    # accurate orthogonalization needed!
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                q = y.dots(y, transp = True)
            s = self.err*self.err - q.reshape((m, 1))
            s[s < 0] = 0
            self.err = numpy.sqrt(s)
            self.eigenvectors.select(sel[1], sel[0])
            self.ncon = ncon
        return self.err


class _DefaultConvergenceCriteria:
    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('res', i)
        return err >= 0 and err <= self.tolerance
        

class _DefaultStoppingCriteria:
    def __init__(self, a, err_tol = 0, sigma_th = 0, max_nsv = 0):
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = _PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.err = self.norms
        #print('max data norm: %e' % numpy.amax(self.err))
        self.err_tol = err_tol
        self.sigma_th = sigma_th
        self.max_nsv = max_nsv
    def satisfied(self, solver):
        self.norms = self.err_calc.norms
        if solver.rcon <= self.ncon:
            return False
        self.err = self.err_calc.update_errors()
        err_rel = numpy.amax(self.err/self.norms)
        lmd = solver.eigenvalues[self.ncon : solver.rcon]
        sigma = -numpy.sort(-numpy.sqrt(abs(lmd)))
        if self.ncon == 0:
            self.sigma = sigma[0]
        now = time.time()
        new = solver.rcon - self.ncon
        elapsed_time = now - self.start_time
        self.elapsed_time += elapsed_time
        i = new - 1
        si = sigma[i]
        si_rel = si/self.sigma
        if self.err_tol <= 0:
            msg = '%.2f sec: sigma[%d] = %e = %.2e*sigma[0], err = %.2e' % \
                (self.elapsed_time, self.ncon + i, si, si_rel, err_rel)
        else:
            print('%.2f sec: sigma[%d] = %.2e*sigma[0], truncation error = %.2e' % \
                  (self.elapsed_time, self.ncon + i, si_rel, err_rel))
        self.ncon = solver.rcon
        if self.err_tol > 0 or self.sigma_th > 0:
            done = err_rel <= self.err_tol or si_rel <= self.sigma_th
        else:
            done = (input(msg + ', more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        done = done or self.max_nsv > 0 and self.ncon >= self.max_nsv
        return done