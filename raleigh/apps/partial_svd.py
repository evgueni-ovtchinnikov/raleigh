# -*- coding: utf-8 -*-
"""Partial SVD of a matrix represented by a 2D ndarray.

Created on Tue Feb 19 13:58:54 2019

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import copy
import math
import numpy
import numpy.linalg as nla
import scipy
import time

from ..solver import Problem, Solver, Options


def truncated_svd(A, opt=Options(), nsv=-1, tol=-1, norm='s', msv=-1, \
                  vtol=1e-3, arch='cpu'):
    '''Computes truncated Singular Value Decomposition of a dense matrix A.
    
    For a given m by n data matrix A computes m by k matrix U, k by k diagonal
    matrix S and n by k matrix V such that A V = U S, the columns of U and V
    are orthonirmal (orthogonal and of unit norm) and the largest singular
    value of A - U S V' is smallest possible for a given k (V' = V.T for a real
    A and A.T.conj() for a complex A).
    The diagonal entries of S are the largest k singular values of A and the
    columns of U and V are corresponding left and right singular vectors.

    Parameters
    ----------
    A : 2D numpy array
        Data matrix.
    opt : an object of class raleigh.solver.Options
        Solver options (see raleigh.solver).
    nsv : int
        Required number of singular values and vectors if known.
        If negative, implicitely defined by the required truncation tolerance
        or interactively by the user.
    tol : float
        Truncation tolerance in the case nsv < 0: if tol > 0, then the norm of
        D = A - U S V' is going to be not greater than the norm of A multiplied 
        by tol, otherwise the user will be asked repeatedly whether the 
        truncation error achieved so far is small enough.
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

    Returns
    -------
    u : numpy array of shape (m, k)
        The matrix U.
    sigma : numpy array of shape (k,)
        The array of the largest k singular values in descending order.
    vt : numpy array of shape (k, n)
        The matrix V'.
    '''
    opt = copy.deepcopy(opt)
    if opt.block_size < 1 and (nsv < 0 or nsv > 100):
        opt.block_size = 128
    if opt.convergence_criteria is None:
        opt.convergence_criteria = _DefaultSVDConvergenceCriteria(vtol)
    if opt.stopping_criteria is None and nsv < 0:
        opt.stopping_criteria = \
            DefaultStoppingCriteria(A, tol, norm, msv)
    psvd = PartialSVD()
    psvd.compute(A, opt, (0, nsv), (None, None), False, False, arch)
    u = psvd.u
    v = psvd.v
    sigma = psvd.sigma
    if msv > 0 and u.shape[1] > msv:
        u = u[:, : msv]
        v = v[:, : msv]
        sigma = sigma[: msv]
    return u, sigma, v.T


def pca(A, opt=Options(), npc=-1, tol=0, norm='f', mpc=0, arch='cpu'):
    '''Performs principal component analysis for the set of data items
    represented by rows of a dense matrix A.

    For a given m by n data matrix A (m data samples n features each)
    computes m by k matrix L and k by n matrix R such that k < min(m, n),
    and the product L R approximates A - e a, where e = numpy.ones((m, 1)
    and a = numpy.mean(A, axis=0).
    The rows of R (principal components) are orhonormal, the columns of L
    (reduced features) are in the descending order of their norms.

    Parameters
    ----------
    A : 2D numpy array
        Data matrix.
    opt : an object of class raleigh.solver.Options
        Solver options (see raleigh.solver).
    npc : int
        Required number of principal components if known.
        If negative, implicitely defined by the required accuracy of
        approximation or interactively by the user.
    tol : float
        Approximation tolerance in the case rank < 0: if tol > 0, then the
        norm of D = A - e a - L R is going to be not greater than the norm
        of A multiplied by tol, otherwise the user will be asked repeatedly
        whether the approximation achieved so far is acceptable.
    norm : character
        The norm to be used for evaluating the approximation error:
        's' : the largest singular value of D,
        'f' : Frobenius norm of D,
        'm' : the largest norm of a row of D.
    mpc : int
        Maximal number of PCs to compute. Ignored if negative, otherwise
        if mpc < min(m, n), then the required accuracy of approximation
        may not be achieved.
    arch : string
        'cpu' : run on CPU,
        'gpu' : run on GPU if available, otherwise on CPU,
        'gpu!' : run on GPU, throw RuntimError if GPU is not present.

    Returns
    -------
    mean : numpy array of shape (1, n)
        The mean of rows of A.
    trans : numpy array of shape (m, k)
        The reduced-features data set.
    comps : numpy array of shape (k, n)
        Principal components.
    '''
    lra = LowerRankApproximation()
    lra.compute(A, opt=opt, rank=npc, tol=tol, norm=norm, \
        max_rank=mpc, shift=True, arch=arch)
    trans = lra.left # transfomed (reduced-features) data
    comps = lra.right # principal components
    return lra.mean, trans, comps


class LowerRankApproximation:
    '''Class for handling the computation of a lower rank approximation of
    a dense matrix, see method compute below for details.
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.mean = None
        self.iterations = -1
    def compute(self, A, opt=Options(), rank=-1, tol=-1, norm='f', max_rank=-1,\
                rtol=1e-3, shift=False, arch='cpu'):
        '''
        For a given m by n data matrix A (m data samples n features each)
        computes m by k matrix L and k by n matrix R such that k < min(m, n),
        and the product L R approximates A if shift=False or else A - e a,
        where e = numpy.ones((A.shape[0], 1) and a = numpy.mean(A, axis=0).
        The rows of R are orhonormal, the columns of L are in the descending
        order of their norms.

        Parameters
        ----------
        A : 2D numpy array
            Data matrix.
        opt : an object of class raleigh.solver.Options
            Solver options (see raleigh.solver).
        rank : int
            Required number of columns in L = number of rows in R (k in the
            above description of the method).
            If negative, implicitely defined by the required accuracy of
            approximation or interactively by the user.
        tol : float
            Approximation tolerance in the case rank < 0: if tol > 0, then the
            norm of the difference D between A (or A - e a) and L R is going
            to be not greater than the norm of A multiplied by tol, otherwise
            the user will be asked repeatedly whether the approximation
            achieved so far is acceptable.
        norm : character
            The norm to be used for evaluating the approximation error:
            's' : the largest singular value of D,
            'f' : Frobenius norm of D,
            'm' : the largest norm of a row of D.
        max_rank : int
            Maximal acceptable rank of L and R. Ignored if negatide, otherwise
            if max_rank < min(m, n), then the required accuracy of approximation
            may not be achieved.
        rtol : float
            Residual tolerance for singular vectors (see Notes below).
            A singular vector is considered converged if the residual 2-norm
            is not greater than rtol multiplied by the largest singular value.
        shift : bool
            Specifies whether L R approximates A (shift=False) or A - e a
            (shift=True, see the above description of the method).
        arch : string
            'cpu' : run on CPU,
            'gpu' : run on GPU if available, otherwise on CPU,
            'gpu!' : run on GPU, throw RuntimError if GPU is not present.

        Notes
        -----
        The rows of R are approximate right singular values of A.
        The columns of L are approximate left singular values of A multiplied
        by respective singular values.
        Singular values and vectors are computed by applying block
        Jacobi-Conjugated Gradient algorithm to A.T A or A A.T, whichever is
        smaller.
        '''
        m, n = A.shape
        opt = copy.deepcopy(opt)
        if opt.block_size < 1:
            opt.block_size = 128
        if opt.convergence_criteria is None:
            opt.convergence_criteria = _DefaultLRAConvergenceCriteria(rtol)
        if opt.stopping_criteria is None and rank < 0:
            opt.stopping_criteria = \
                DefaultStoppingCriteria(A, tol, norm, max_rank)
        psvd = PartialSVD()
        psvd.compute(A, opt, nsv=(0, rank), shift=shift, arch=arch)
        self.left = psvd.sigma*psvd.u # left multiplier L
        self.right = psvd.v.T # right multiplier R
        self.mean = psvd.mean # bias
        if max_rank > 0 and self.left.shape[1] > max_rank:
            self.left = self.left[:, : max_rank]
            self.right = self.right[: max_rank, :]
        self.iterations = psvd.iterations


class PartialSVD:
    def __init__(self):
        self.sigma = None
        self.u = None
        self.v = None
        self.mean = None
        self.iterations = -1
    def compute(self, a, opt, nsv=(-1, -1), isv=(None, None), \
                shift=False, refine=False, arch='cpu'):
    
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
            from ..ndarray.algebra import Vectors, Matrix
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
                    tmp = Vectors(n, l, data_type=dt)
                    op.apply(isv[i], tmp)
                    isvec += (tmp,)
                else:
                    isvec += (None,)
            isv = isvec
    
        v = Vectors(n, data_type=dt)
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
    
        solver.solve(v, opt, which=nsv, init=isv)
        if opt.verbosity > 0:
            print('operator application time: %.2e' % opSVD.time)
    
        nv = v.nvec()
        u = Vectors(m, nv, v.data_type())
        if nv > 0:
            op.apply(v, u, transp)
            if shift:
                m, n = a.shape
                dt = op.data_type()
                ones = numpy.ones((1, m), dtype=dt)
                e = Vectors(m, 1, data_type=dt)
                e.fill(ones)
                w = Vectors(n, 1, data_type=dt)
                op.apply(e, w, transp=True)
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
                lmd, x = scipy.linalg.eigh(uu, vv) #, turbo=False)
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
            sigma = numpy.ndarray((0,), dtype=v.data_type())
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


class PSVDErrorCalculator:
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
        self.norms = _norm(a, axis=1).reshape((m, 1))
        self.err = self.norms.copy()
        self.aves = None
    def set_up(self, op, solver, eigenvectors, shift=False):
        self.op = op
        self.solver = solver
        self.eigenvectors = eigenvectors
        self.shift = shift
        if shift:
            self.ones = eigenvectors.new_vectors(1, self.m)
            ones = numpy.ones((1, self.m), dtype=self.dt)
            self.ones.fill(ones)
            self.aves = eigenvectors.new_vectors(1, self.n)
            self.op.apply(self.ones, self.aves, transp=True)
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
            s = self.err*self.err - q.reshape((m, 1))
            s[s < 0] = 0
            self.err = numpy.sqrt(s)
            self.eigenvectors.select(sel[1], sel[0])
            self.ncon = ncon
        return self.err


class DefaultStoppingCriteria:
    def __init__(self, a, err_tol=0, norm='f', max_nsv=0):
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.err_tol = err_tol
        self.max_nsv = max_nsv
        self.norm = norm
        self.f = 0
    def satisfied(self, solver):
        self.norms = self.err_calc.norms
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
            err_rel = numpy.amax(self.err)/numpy.amax(self.norms)
        elif self.norm == 'f':
            self.f -= numpy.sum(sigma*sigma)
            err_rel = math.sqrt(abs(self.f)/numpy.sum(self.norms*self.norms))
        else:
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
        if self.err_tol > 0:
            print(msg)
            done = err_rel <= self.err_tol
        else:
            done = (input(msg + ', more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        done = done or self.max_nsv > 0 and self.ncon >= self.max_nsv
        return done


class _OperatorSVD:
    def __init__(self, op, v, gpu, transp=False, shift=False):
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


class _DefaultSVDConvergenceCriteria:
    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('kinematic vector error', i)
        return err >= 0 and err <= self.tolerance


class _DefaultLRAConvergenceCriteria:
    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('residual', i)
        return err >= 0 and err <= self.tolerance


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a
