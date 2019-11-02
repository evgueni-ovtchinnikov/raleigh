# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""LowerRankApproximation of a matrix.
"""

import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla

from ..core.solver import Options
from .partial_svd import AMatrix
from .partial_svd import PartialSVD
from .partial_svd import DefaultStoppingCriteria


class LowerRankApproximation:
    '''Class for handling the computation of a lower rank approximation of
    a dense matrix, see method compute below for details.
    '''
    def __init__(self, mlr=None):
        if mlr is None:
            self.__left = None
            self.__right = None
            self.__mean = None
            self.__rank = 0
            self.__dtype = None
        else:
            self.__mean = mlr[0]
            self.__left = mlr[1]
            self.__right = mlr[2]
            self.__rank = self.__right.shape[0]
            self.__dtype = self.__left.dtype.type
        self.__left_v = None
        self.__right_v = None
        self.__mean_v = None
        self.__tol = -1
        self.__svtol = 1e-3
        self.__norm = None
        self.__arch = None
        self.ortho = True
        self.iterations = -1
        
    def compute(self, data_matrix, opt=Options(), rank=-1, tol=0, norm='f',
                max_rank=-1, svtol=1e-3, shift=False, verb=0):
        '''
        For a given m by n data matrix A (m data samples n features each)
        computes m by k matrix L and k by n matrix R such that k <= min(m, n),
        and the product L R approximates A if shift is False or else A - e a,
        where e = numpy.ones((m, 1)) and a = numpy.mean(A, axis=0).

        The rows of R are orhonormal, the columns of L are in the descending
        order of their norms.

        Below A_ stands for A if shift is False and for A - e a otherwise.

        Parameters
        ----------
        data_matrix : AMatrix
            Data matrix A.
        opt : an object of class raleigh.solver.Options
            Solver options (see raleigh.solver).
        rank : int
            Required number of columns in L = number of rows in R (k in the
            above description of the method).
            If negative, implicitely defined by the required accuracy of
            approximation or interactively by the user.
        tol : float
            Approximation tolerance in the case rank < 0. If tol is non-zero, 
            then the computation of the lower rank approximation will stop
            when the norm of the difference D = A_ - L R becomes not greater
            than eps, where eps is the norm of A multiplied by tol if tol > 0
            and eps = -tol if tol < 0. Otherwise the user will be asked 
            repeatedly whether the computation should continue (the number of
            computed principal components and the relative truncation error, 
            the ratio of the norm of D to that of A, are displayed).
        norm : character
            The norm to be used for evaluating the approximation error:
            's' : the largest singular value of D (if tol > 0, then the largest
                  singular value of A - e a will be used instead of the norm of
                  A for evaluating the relative truncation error),
            'f' : Frobenius norm of D,
            'm' : the largest norm of a row of D.
        max_rank : int
            Maximal acceptable rank of L and R. Ignored if negatide, otherwise
            if max_rank < min(m, n), then the required accuracy of approximation
            might not be achieved.
        svtol : float
            Error tolerance for singular values (see Notes below) relative to
            the largest singular value.
        shift : bool
            Specifies whether L R approximates A (shift=False) or A_ = A - e a
            (shift=True, see the above description of the method).
        verb : int
            Verbosity level.

        Notes
        -----
        The rows of R are approximate right singular vectors of A_.
        The columns of L are approximate left singular vectors of A_ multiplied
        by respective singular values.
        Singular values and vectors are computed by applying block
        Jacobi-Conjugated Gradient algorithm to A_.T A_ or A_ A_.T, whichever
        is smaller in size.
        '''
        m, n = data_matrix.shape()
        user_bs = opt.block_size
        if user_bs < 1 and (rank < 0 or rank > 100):
            opt.block_size = 128
        if opt.convergence_criteria is None:
            no_cc = True
            opt.convergence_criteria = _DefaultLRAConvergenceCriteria(svtol)
        else:
            no_cc = False
        if opt.stopping_criteria is None and rank < 0:
            no_sc = True
            opt.stopping_criteria = \
                DefaultStoppingCriteria(data_matrix, tol, norm, max_rank, verb)
        else:
            no_sc = False

        psvd = PartialSVD()
        psvd.compute(data_matrix, opt=opt, nsv=(0, rank), shift=shift, \
            refine=self.ortho)
        self.__left_v = psvd.left_v()
        self.__left_v.scale(psvd.sigma, multiply=True)
        self.__right_v = psvd.right_v()
        self.__mean_v = psvd.mean_v()
        self.__rank = self.__left_v.nvec()
        self.__opt = opt
        self.__tol = tol
        self.__svtol = svtol
        self.__norm = norm
        self.__arch = data_matrix.arch()
        self.__dtype = data_matrix.data_type()
        if max_rank > 0 and self.__left_v.nvec() > max_rank:
            self.__left_v.select(max_rank)
            self.__right_v.select(max_rank)
        self.iterations = psvd.iterations
        
        # restore user opt to avoid side effects
        opt.block_size = user_bs
        if no_cc:
            opt.convergence_criteria = None
        if no_sc:
            opt.stopping_criteria = None

    def update(self, data_matrix, opt=None, rank=-1, max_rank=-1, \
            tol=None, norm=None, svtol=None, verb=0):
        '''
        Updates previously computed Lower Rank Approximation of a data matrix
        data_matrix0, i.e. computes Lower Rank Approximation of 
        numpy.concatenate((data_matrix0, data_matrix))

        Parameters have the same meaning as for compute().
        '''
        if self.__rank == 0:
            raise RuntimeError('no existing LRA data to update')
        if opt is None:
            opt = self.__opt
        if tol is None:
            tol = self.__tol
        if tol == 0.0 and rank < 1:
            rank = self.__rank
        if norm is None:
            norm = self.__norm
        if svtol is None:
            svtol = self.__svtol
        if norm not in ['f', 'm', 's']:
            msg = 'norm %s is not supported' % repr(norm)
            raise ValueError(msg)
        v = data_matrix.as_vectors()
        s = abs(v.dots(v))
        fnorm = math.sqrt(numpy.sum(s))
        maxl2norm = numpy.amax(numpy.sqrt(s))
        if maxl2norm == 0.0:
            return
        dtype = self.__dtype
        if self.__left_v is None:
            left_data = self.__left.T
            # TODO: this must be taken care of by new_vectors()
            if not left_data.flags['C_CONTIGUOUS']:
                left_data = numpy.ndarray(left_data.shape, dtype=dtype)
                left_data[:,:] = self.__left.T.copy()
            self.__left_v = v.new_vectors(left_data)
            self.__right_v = v.new_vectors(self.__right)
            if self.__mean is not None:
                self.__mean_v = v.new_vectors(self.__mean)
            else:
                self.__mean_v = None
            self.__arch = data_matrix.arch()
        else:
            if self.__arch != data_matrix.arch() or \
                dtype != data_matrix.data_type():
                raise ValueError('incompatible matrix type passed to update')
        left0 = self.__left_v
        right0 = self.__right_v
        if self.ortho is False:
            wl = left0.new_vectors(left0.nvec())
            wr = right0.new_vectors(right0.nvec())
            H = right0.dot(right0)
            mu, x = sla.eigh(H)
            q = mu[0]/mu[-1]
            print(mu[0], mu[-1])
            if q < 0.5:
                _lra_ortho(left0, right0, wl, wr)
            else:
                G = left0.dot(left0)
                lmd, x = sla.eigh(-G, H)
                y = nla.inv(x.T)
                left0.multiply(y, wl)
                wl.copy(left0)
                right0.multiply(x, wr)
                wr.copy(right0)
        shift = self.__mean_v is not None
        sigma = numpy.sqrt(left0.dots(left0))
        sigma0 = sigma[0]
        n0 = left0.dimension()
        e0 = numpy.ones((n0, 1), dtype=dtype)
        n1 = v.nvec()
        e1 = numpy.ones((n1, 1), dtype=dtype)
        n = n0 + n1

        if shift:
            mean0 = self.__mean_v.data()
            mean1 = v.new_vectors(1, v.dimension())
            v.multiply(e1, mean1)
            mean1 = mean1.data()/n1
            mean = (n0/n)*mean0 + (n1/n)*mean1
            diff = mean0 - mean
            vdiff = v.new_vectors(diff)
            vdiff0 = vdiff.orthogonalize(right0)
            diff0 = vdiff0.data().T
            s = nla.norm(vdiff.data())*e0[:1]
            vdiff.scale(s)
            e0v = v.new_vectors(e0.T)
            left0.add(e0v, 1.0, diff0)
            e0v.scale(s, multiply=True)
            left0.append(e0v)
            right0.append(vdiff)
            vmean = v.new_vectors(mean)
            v.add(vmean, -1.0, e1.T)
        else:
            mean = None
            vmean = None

        s = abs(v.dots(v))
        fnorm = math.sqrt(numpy.sum(s))
        maxl2norm = numpy.amax(numpy.sqrt(s))

        left1 = v.orthogonalize(right0)

        lra = LowerRankApproximation()
        if rank < 0:
            if norm == 'f':
                update_tol = -tol*fnorm
            elif norm == 'm':
                update_tol = -tol*maxl2norm
            else:
                update_tol = -tol*sigma0
            urank = max_rank*n1//(n0 + n1)
            lra.compute(data_matrix, opt, tol=update_tol, norm=norm, \
                max_rank=urank, verb=verb)
        else:
            urank = rank*n1//(n0 + n1)
            if verb > 0:
                print('computing new %d components...' % urank)
            lra.compute(data_matrix, opt, rank=urank, verb=verb)

        left11 = lra.left_v()
        right10 = lra.right_v()
        
        new = left11.nvec()
        left01 = left0.new_vectors(new)
        left01.zero()
        left0.append(left01)
        left1.append(left11)
        left0_data = left0.data()
        left1_data = left1.data()
        left0_data = numpy.concatenate((left0_data, left1_data), axis=1)
        left0 = left0.new_vectors(left0_data)
#        left0.append(left1, axis=1)
        right0.append(right10)
        self.__left_v = left0
        self.__right_v = right0

        wl = left0.new_vectors(left0.nvec())
        wr = right0.new_vectors(right0.nvec())
        H = right0.dot(right0)
        mu, x = sla.eigh(H)
        q = mu[0] #/mu[-1]
        if q < 0.5:
#            k = numpy.sum(mu < 0.5)
#            print(mu[: k + 1])
            _lra_ortho(left0, right0, wl, wr)
        else:
            G = left0.dot(left0)
            lmd, x = sla.eigh(-G, H)
            y = nla.inv(x.T)
            left0.multiply(y, wl)
            wl.copy(left0)
            right0.multiply(x, wr)
            wr.copy(right0)

        if rank < 0:
            ncomp = right0.nvec()
            e = numpy.ones((n, 1), dtype=dtype)
            if norm == 'f':
                r = left0.dots(left0)
                s = math.sqrt(numpy.sum(r))
            elif norm == 'm':
                r = left0.dots(left0, transp=True)
                s = numpy.amax(numpy.sqrt(abs(r)))
            else:
                s = sigma[0]
            if shift and False:
                a = vmean.dot(vmean)
                b = vmean.dot(right0)
                p = left0.new_vectors(1)
                left0.multiply(b, p)
                if norm == 'f':
                    q = left0.new_vectors(e.T)
                    c = q.dot(p)
                    s = math.sqrt(s*s + 2*c + n*a)
                elif norm == 'm':
                    p = p.data()
                    s = math.sqrt(numpy.amax(r + 2*p.T) + a)
            eps = s*tol/4
            if norm == 'm':
                errs = numpy.zeros((1, n))
            s = 0
            i = 1
            while i < ncomp:
                if norm == 'f':
                    s = math.sqrt(s*s + r[ncomp - i])
                elif norm == 'm':
                    left0.select(1, ncomp - i)
                    lft = left0.data()
                    errs += lft * lft
                    s = numpy.amax(numpy.sqrt(errs))
                else:
                    s = sigma[ncomp - i]
                if s > eps:
                    break
                i += 1
            i -= 1
            if i > 0:
                print('discarding %d components out of %d' % (i, ncomp))
                ncomp -= i
        else:
            ncomp = rank

        left0.select(ncomp)
        right0.select(ncomp)
        self.__left = None
        self.__right = None
        self.__mean = None
        if shift:
            self.__mean_v = vmean
        self.__rank = self.__left_v.nvec()
        self.__tol = tol
        self.__svtol = svtol
        self.__norm = norm
        self.__arch = data_matrix.arch()
        self.__dtype = data_matrix.data_type()
        if max_rank > 0 and self.__left_v.nvec() > max_rank:
            self.__left_v.select(max_rank)
            self.__right_v.select(max_rank)
        self.iterations += lra.iterations

    def icompute(self, data_matrix, batch_size, opt=Options(), rank=-1, \
                 tol=0, norm='f', max_rank=-1, svtol=1e-3, shift=False, \
                 arch='cpu', verb=0):
        '''
        Computes Lower Rank Approximation of data_matix incrementally:
            - computes LRA for the block data_matrix[:batch_size, :]
            - in a loop updates LRA using subsequent blocks of data_matrix
              batch_size rows high (or less, for the last block)

        Parameters
        ----------
        data_matrix : 2D numpy array
            Data matrix A.
        batch_size : int
            The batch size.

        Remaining parameters have the same meaning as for compute().
        '''
        data_size = data_matrix.shape[0]
        batch_size = min(batch_size, data_size)
        batch = 0
        if self.__rank == 0:
            print('processing batch %d of size %d' % (batch, batch_size))
            matrix = AMatrix(data_matrix[:batch_size, :], arch=arch)
            self.compute(matrix, opt=opt, rank=rank, \
                         tol=tol, norm=norm, max_rank=max_rank, svtol=svtol, \
                         shift=shift, verb=verb)
            first = batch_size
            batch += 1
        else:
            first = 0
        while first < data_size:
            next_ = min(data_size, first + batch_size)
            print('processing batch %d of size %d' % (batch, next_ - first))
            matrix = AMatrix(data_matrix[first : next_, :], arch=arch, \
                             copy_data=True)
            self.update(matrix, opt=opt, rank=rank, tol=tol, norm=norm, \
                        max_rank=max_rank, svtol=svtol, verb=verb)
            first = next_
            batch += 1

    def mean(self): # mean row of A
        if self.__mean is None:
            if self.__mean_v is not None:
                self.__mean = self.__mean_v.data()
        return self.__mean

    def left(self): # left multiplier L
        if self.__left is None:
            if self.__left_v is not None:
                self.__left = self.__left_v.data().T
        return self.__left

    def right(self): # right multiplier R
        if self.__right is None:
            if self.__right_v is not None:
                self.__right = self.__right_v.data()
        return self.__right

    def mean_v(self):
        return self.__mean_v

    def left_v(self):
        return self.__left_v

    def right_v(self):
        return self.__right_v


class _DefaultLRAConvergenceCriteria:

    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        res = solver.convergence_data('residual', i)
        lmd = solver.convergence_data('eigenvalue', i)
        lmd_max = solver.convergence_data('max eigenvalue', i)
        tol = abs(lmd/lmd_max)**1.5*self.tolerance
        return res >= 0 and res*res <= tol


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a


def _lra_ortho(v, u, wv, wu):

    u.copy(wu)
    s, q = wu.svd()
    v.multiply(_conj(q.T), wv)
    wv.scale(s, multiply=True)
    wv.copy(v)
    s, q = v.svd()
    wu.multiply(_conj(q.T), u)
    v.scale(s, multiply=True)
