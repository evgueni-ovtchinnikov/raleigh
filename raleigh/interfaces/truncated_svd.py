# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""Truncated Singular Value Decomposition of a matrix 
   represented by a 2D ndarray.
"""

import math
import numpy
import numpy.linalg as nla
import time

from ..core.solver import Options
from ..algebra.dense_matrix import AMatrix
from .partial_svd import PartialSVD

try:
    input = raw_input
except NameError:
    pass


def truncated_svd(matrix, opt=Options(), nsv=-1, tol=-1, norm='s', msv=-1, \
                  vtol=1e-3, arch='cpu', verb=0):
    '''Computes truncated Singular Value Decomposition of a dense matrix A.
    
    For a given m by n matrix A computes m by k matrix U, k by k diagonal
    matrix S and n by k matrix V such that A V = U S, the columns of U and V
    are orthonirmal (orthogonal and of unit norm) and the largest singular
    value of A - U S V' is smallest possible for a given k (V' = V.T for a real
    A and A.T.conj() for a complex A).
    The diagonal entries of S are the largest k singular values of A and the
    columns of U and V are corresponding left and right singular vectors.

    Parameters
    ----------
    matrix : 2D numpy array
        Matrix A.
    opt : an object of class raleigh.solver.Options
        Solver options (see raleigh.solver).
    nsv : int
        Required number of singular values and vectors if known.
        If negative, implicitely defined by the required truncation tolerance
        or interactively by the user.
    tol : float
        Truncation tolerance in the case nsv < 0. If tol is non-zero, then the 
        computation of singular values and vectors will stop when the norm of 
        D = A - U S V' becomes not greater than eps, where eps is the norm of 
        A multiplied by tol if tol > 0 and eps = -tol if tol < 0. If tol is 
        zero, then the user will be asked repeatedly whether the computation
        should continue (the number of computed singular values and the 
        relative truncation error, the ratio of the norm of D to that of A, are
        displayed).
    norm : character
        The norm to be used for evaluating the truncation error:
        's' : the largest singular value of D,
        'f' : Frobenius norm of D,
        'm' : the largest norm of a row of D.
    msv : int
        Maximal number of singular values to compute. Ignored if negative, 
        otherwise if msv < min(m, n), then the truncation error may be greater
        than requested.
    vtol : float
        Singular vectors error tolerance.
    arch : string
        'cpu' : run on CPU,
        'gpu' : run on GPU if available, otherwise on CPU,
        'gpu!' : run on GPU, throw RuntimError if GPU is not present.
    verb : int
        Verbosity level.

    Returns
    -------
    u : numpy array of shape (m, k)
        The matrix U.
    sigma : numpy array of shape (k,)
        The array of the largest k singular values in descending order.
    vt : numpy array of shape (k, n)
        The matrix V'.

    Notes
    -----
    This solver can only be efficient if singular values decay quickly, e.g.
    exponentially. If the singular values properties are unknown, then it is
    worth to try it first in interactive mode (nsv < 0, tol = 0), and if the
    computation is too slow, use scipy.linalg.svd instead.
    '''
    matrix = AMatrix(matrix, arch=arch)
    psvd = PartialSVD(matrix)

    user_bs = opt.block_size
    if user_bs < 1 and (nsv < 0 or nsv > 100):
        opt.block_size = 128
    if opt.convergence_criteria is None:
        no_cc = True
        opt.convergence_criteria = _DefaultSVDConvergenceCriteria(vtol)
    else:
        no_cc = False
    if opt.stopping_criteria is None and nsv < 0:
        no_sc = True
        opt.stopping_criteria = \
            DefaultStoppingCriteria(matrix, tol, norm, msv, verb)
        v = psvd.vectors()
        opSVD = psvd.op_svd()
        opt.stopping_criteria.err_calc.set_up(opSVD, v, shift=False)
    else:
        no_sc = False

    psvd.compute(matrix, opt, nsv=(0, nsv))
    u = psvd.left()
    v = psvd.right()
    sigma = psvd.sigma
    if msv > 0 and u.shape[1] > msv:
        u = u[:, : msv]
        v = v[:, : msv]
        sigma = sigma[: msv]

    # restore user opt to avoid side effects
    opt.block_size = user_bs
    if no_cc:
        opt.convergence_criteria = None
    if no_sc:
        opt.stopping_criteria = None

    return u, sigma, v.T


class TruncatedSVDErrorCalculator:
    def __init__(self, a):
        m, n = a.shape()
        self.dt = a.data_type()
        s = a.dots()
        self.norms = numpy.sqrt(s.reshape((m, 1)))
        self.solver = None
        self.err = None
        self.op = None
        self.m = m
        self.n = n
        self.shift = False
        self.ncon = 0
        self.err = self.norms.copy()
        self.aves = None
    def set_up(self, op, eigenvectors, shift=False):
        self.op = op.op
        self.eigenvectors = eigenvectors
        self.shift = shift
        if shift:
            self.ones = op.ones
            self.aves = op.aves
            s = self.aves.dots(self.aves)
            vb = eigenvectors.new_vectors(1, self.m)
            self.op.apply(self.aves, vb)
            b = vb.data().reshape((self.m, 1))
            t = (self.norms*self.norms).reshape((self.m, 1))
            x = t - 2*b + s*numpy.ones((self.m, 1))
            self.err = numpy.sqrt(abs(x))
        self.err_init = numpy.amax(self.err)
        self.err_init_f = nla.norm(self.err)
    def update_errors(self):
        ncon = self.eigenvectors.nvec()
        new = ncon - self.ncon
        if new > 0:
            err = self.err*self.err 
            x = self.eigenvectors
            sel = x.selected()
            x.select(new, self.ncon)
            m = self.m
            n = self.n
            if m < n:
                z = x.new_vectors(new, n)
                self.op.apply(x, z, transp=True)
                if self.shift:
                    s = x.dot(self.ones)
                    z.add(self.aves, -1, s)
                y = x.new_vectors(new, m)
                self.op.apply(z, y)
                if self.shift:
                    s = z.dot(self.aves)
                    y.add(self.ones, -1, s)
                q = x.dots(y, transp=True)
                q[q < 0] = 0
                err[q <= 0] = 0
            else:
                y = x.new_vectors(new, m)
                self.op.apply(x, y)
                if self.shift:
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                    # accurate orthogonalization needed!
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                q = y.dots(y, transp=True)
            q = q.reshape((m, 1))
            err -= q
            err[err < 0] = 0
            self.err = numpy.sqrt(err)
            self.eigenvectors.select(sel[1], sel[0])
            self.ncon = ncon
        return self.err


class DefaultStoppingCriteria:

    def __init__(self, a, err_tol=0, norm='f', max_nsv=0, verb=0):
        self.shape = a.shape()
        self.scale = a.scale()
        self.err_tol = err_tol
        self.norm = norm
        self.max_nsv = max_nsv
        self.verb = verb
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = TruncatedSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.max_norm = numpy.amax(self.norms)
        self.f_norm = math.sqrt(numpy.sum(self.norms*self.norms))
        self.f = 0

    def satisfied(self, solver):
        self.norms = self.err_calc.norms
        m, n = self.shape
#        scale_max = self.scale*math.sqrt(n)
#        scale_f = self.scale*math.sqrt(m*n)
        scale_max = self.err_calc.err_init
        scale_f = self.err_calc.err_init_f
        if solver.rcon <= self.ncon:
            return False
        new = solver.rcon - self.ncon
        lmd = solver.eigenvalues[self.ncon : solver.rcon]
        sigma = -numpy.sort(-numpy.sqrt(abs(lmd)))
        if self.ncon == 0:
            self.sigma = sigma[0]
            self.err = self.err_calc.err
            self.f = numpy.sum(self.err*self.err)
        i = new - 1
        si = sigma[i]
        si_rel = si/self.sigma
        if self.norm == 'm':
            self.err = self.err_calc.update_errors()
            err_abs = numpy.amax(self.err)
            err_rel = err_abs/scale_max
#            err_rel = err_abs/self.max_norm
        elif self.norm == 'f':
            self.f -= numpy.sum(sigma*sigma)
            err_abs = math.sqrt(abs(self.f))
            err_rel = err_abs/scale_f
#            err_rel = err_abs/self.f_norm
        else:
            err_abs = si
            err_rel = si_rel
        now = time.time()
        elapsed_time = now - self.start_time
        self.elapsed_time += elapsed_time
        if self.norm in ['f', 'm']:
            msg = '%.2f sec: sigma[%d] = %.2e*sigma[0], truncation error = %.2e' % \
                  (self.elapsed_time, self.ncon + i, si_rel, err_rel)
        else:
            msg = '%.2f sec: sigma[%d] = %e = %.2e*sigma[0]' % \
                (self.elapsed_time, self.ncon + i, si, si_rel)
        self.ncon = solver.rcon
        done = False
        if self.err_tol != 0:
            if self.verb > 0:
                print(msg)
            if self.err_tol > 0:
                done = err_rel <= self.err_tol
            else:
                done = err_abs <= abs(self.err_tol)
        elif self.max_nsv < 1:
            done = (input(msg + ', more? ') == 'n')
        elif self.verb > 0:
            print(msg)
        self.iteration = solver.iteration
        self.start_time = time.time()
        done = done or self.max_nsv > 0 and self.ncon >= self.max_nsv
        return done


class _DefaultSVDConvergenceCriteria:
    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('kinematic vector error', i)
        return err >= 0 and err <= self.tolerance
