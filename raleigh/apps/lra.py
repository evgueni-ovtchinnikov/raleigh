# -*- coding: utf-8 -*-
"""Lower Rank Approximation of a dense matrix

Created on Tue Feb 19 14:07:09 2019

@author: Evgueni Ovtchinnikov, UKRI
"""

import copy

from .partial_svd import DefaultStoppingCriteria
from .partial_svd import PartialSVD
from ..solver import Options

def pca(a, opt = Options(), npc = -1, tol = 0, th = 0, msv = 0, ipc = None, \
        arch = 'cpu'):
    lra = LowerRankApproximation()
    lra.compute(a, opt = opt, nsv = npc, tol = tol, th = th, \
        msv = msv, isv = ipc, shift = True, arch = arch)
    return lra.mean, lra.u, lra.sigma, lra.v.T


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
            opt.stopping_criteria = DefaultStoppingCriteria(a, tol, th, msv)
        psvd = PartialSVD()
        psvd.compute(a, opt, (0, nsv), (None, isv), shift, refine, arch)
        self.sigma = psvd.sigma
        self.u = psvd.u
        self.v = psvd.v
        self.mean = psvd.mean
        self.iterations = psvd.iterations
#        return self.sigma, self.u, self.v.T


class _DefaultConvergenceCriteria:
    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('res', i)
        return err >= 0 and err <= self.tolerance