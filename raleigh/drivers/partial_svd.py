# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Partial SVD of a matrix represented by a 2D ndarray.
"""

import copy
import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla
import time

try:
    input = raw_input
except NameError:
    pass

from ..core.solver import Problem, Solver, Options


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
    opt = copy.deepcopy(opt)
    if opt.block_size < 1 and (nsv < 0 or nsv > 100):
        opt.block_size = 128
    if opt.convergence_criteria is None:
        opt.convergence_criteria = _DefaultSVDConvergenceCriteria(vtol)
    matrix = AMatrix(matrix, arch=arch)
    if opt.stopping_criteria is None and nsv < 0:
        opt.stopping_criteria = \
            DefaultStoppingCriteria(matrix, tol, norm, msv, verb)
    psvd = PartialSVD()
    psvd.compute(matrix, opt, nsv=(0, nsv))
    u = psvd.left()
    v = psvd.right()
    sigma = psvd.sigma
    if msv > 0 and u.shape[1] > msv:
        u = u[:, : msv]
        v = v[:, : msv]
        sigma = sigma[: msv]
    return u, sigma, v.T


def pca(A, opt=Options(), npc=-1, tol=0, norm='f', mpc=-1, svtol=1e-3, \
        have=None, batch_size=None, arch='cpu', verb=0):
    '''Performs principal component analysis for the set of data items
    represented by rows of a dense matrix A.

    For a given m by n data matrix A (m data samples n features each)
    computes m by k matrix L and k by n matrix R such that k <= min(m, n),
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
        Approximation tolerance in the case rank < 0. If tol is non-zero, then 
        the computation of principal components will stop when the norm of 
        D = A - e a - L R becomes not greater than eps, where eps is the norm
        of A multiplied by tol if tol > 0 and eps = -tol if tol < 0. Otherwise
        the user will be asked repeatedly whether the computation should 
        continue (the number of computed principal components and the relative 
        truncation error, the ratio of the norm of D to that of A, are 
        displayed).
    norm : character
        The norm to be used for evaluating the approximation error:
        's' : the largest singular value of D (if tol > 0, then the largest
              singular value of A - e a will be used instead of the norm of A
              for evaluating the relative truncation error),
        'f' : Frobenius norm of D,
        'm' : the largest norm of a row of D.
    mpc : int
        Maximal number of PCs to compute. Ignored if negative, otherwise
        if mpc < min(m, n), then the required accuracy of approximation
        might not be achieved.
    svtol : float
        Error tolerance for singular values (see Notes below) relative to
        the largest singular value.
    have : tuple (a0, L0, R0)
        If not None, previously computed PCA approximation L0 R0 + e0 a0
        to a data matrix A0 of the same width as A is to be updated, i.e.
        PCA approximation L R + e a to numpy.concatenate((A0, A)) is to be
        computed.
    arch : string
        'cpu' : run on CPU,
        'gpu' : run on GPU if available, otherwise on CPU,
        'gpu!' : run on GPU, throw RuntimError if GPU is not present.
    verb : int
        Verbosity level.

    Returns
    -------
    mean : numpy array of shape (1, n)
        The mean of rows of A.
    trans : numpy array of shape (m, k)
        The reduced-features data matrix.
    comps : numpy array of shape (k, n)
        Principal components matrix.

    Notes
    -----
    If have is None, then the rows of R are approximate right singular
    vectors of A - e a and the columns of L are approximate left singular
    vectors of A - e a multiplied by respective singular values. If have is
    not None, this is generally not the case.
    '''
    lra = LowerRankApproximation(have)
    if batch_size is None:
        if have is None:
            data_matrix = AMatrix(A, arch=arch)
            lra.compute(data_matrix, opt=opt, rank=npc, tol=tol, norm=norm, \
                        max_rank=mpc, svtol=svtol, shift=True, verb=verb)
        else:
            data_matrix = AMatrix(A, arch=arch, copy_data=True)
            lra.update(data_matrix, opt=opt, rank=npc, tol=tol, norm=norm, \
                       max_rank=mpc, svtol=svtol)
    else:
        lra.icompute(A, batch_size, opt=opt, rank=npc, tol=tol, norm=norm, \
                        max_rank=mpc, svtol=svtol, shift=True, arch=arch, \
                        verb=verb)
    trans = lra.left() # transfomed (reduced-features) data
    comps = lra.right() # principal components
    return lra.mean(), trans, comps


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
        
    def compute(self, data_matrix, opt=Options(), rank=-1, tol=-1, norm='f',
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
        if user_bs < 1:
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
        if rank > 0 and rank <= self.__rank:
            return
        if max_rank > 0 and self.__rank <= max_rank:
            return
        if opt is None:
            opt = self.__opt
        if tol is None:
            tol = self.__tol
        if norm is None:
            norm = self.__norm
        if svtol is None:
            svtol = self.__svtol
        if norm not in ['f', 'm', 's']:
            msg = 'norm %s is not supported' % repr(norm)
            raise ValueError(msg)
        v = data_matrix.as_vectors() # reference to data_matrix
        s = abs(v.dots(v))
        fnorm = math.sqrt(numpy.sum(s))
        maxl2norm = numpy.amax(numpy.sqrt(s))
        if maxl2norm == 0.0:
            return
        dtype = self.__dtype
        if self.__left_v is None:
            left_data = self.__left.T
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
            G = left0.dot(left0)
            H = right0.dot(right0)
            lmd, x = sla.eigh(-G, H)
            left0.multiply(nla.inv(x.T), wl)
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

        left1 = v.orthogonalize(right0)
        if norm == 'f':
            update_tol = -tol*fnorm
        elif norm == 'm':
            update_tol = -tol*maxl2norm
        else:
            update_tol = -tol*sigma0

        s = abs(v.dots(v))
        fnorm = math.sqrt(numpy.sum(s))
        maxl2norm = numpy.amax(numpy.sqrt(s))
        
        lra = LowerRankApproximation()
        lra.compute(data_matrix, opt, tol=update_tol, norm=norm, verb=verb)
        left11 = lra.left_v()
        right10 = lra.right_v()
        
        new = left11.nvec()
        left01 = left0.new_vectors(new)
        left01.zero()
        left0.append(left01)
        left1.append(left11)
        left0.append(left1, axis=1)
        right0.append(right10)

        wl = left0.new_vectors(left0.nvec())
        wr = right0.new_vectors(right0.nvec())
        G = left0.dot(left0)
        H = right0.dot(right0)
        lmd, x = sla.eigh(-G, H)
        sigma = numpy.sqrt(abs(lmd))
        left0.multiply(nla.inv(x.T), wl)
        wl.copy(left0)
        right0.multiply(x, wr)
        wr.copy(right0)
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
        if shift:
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
        #print(s)
        lmd = sigma*sigma
        eps = s*tol/7
        if norm == 'm':
            errs = numpy.zeros((1, n))
        s = 0
        i = 1
        while s < eps and i < ncomp:
            if norm == 'f':
                s = math.sqrt(s*s + lmd[ncomp - i])
            elif norm == 'm':
                left0.select(1, ncomp - i)
                lft = left0.data()
                errs += lft * lft
                s = numpy.amax(numpy.sqrt(errs))
            else:
                s = sigma[ncomp - i]
            i += 1
        print('discarding %d components out of %d' % (i, ncomp))
        ncomp -= i
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
                 tol=-1, norm='f', max_rank=-1, svtol=1e-3, shift=False, \
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
            del matrix # free memory
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
            del matrix # free memory
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


''' The rest is for advanced users only.
'''
class PartialSVD:

    def __init__(self):
        self.sigma = None
        self.__left = None
        self.__right = None
        self.__mean = None
        self.__left_v = None
        self.__right_v = None
        self.__mean_v = None
        self.iterations = -1

    def compute(self, matrix, opt=Options(), nsv=(-1, -1), shift=False, \
                refine=True):
    
        op = matrix.as_operator()
        m, n = matrix.shape()
    
        transp = m < n
        if transp:
            n, m = m, n

        v = op.new_vectors(n)
        dt = v.data_type()
        opSVD = _OperatorSVD(matrix, v, transp, shift)
        problem = Problem(v, opSVD)
        solver = Solver(problem)

        try:
            opt.stopping_criteria.err_calc.set_up(opSVD, solver, v, shift)
            if opt.verbosity > 0:
                print('partial SVD error calculation set up')
        except:
            if opt.verbosity > 0:
                print('partial SVD error calculation not requested')
            pass

        solver.solve(v, options=opt, which=nsv)
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
                u_data = u.data()
                uu, sigma, uv = sla.svd(u_data.T, full_matrices=False)
                u = u.new_vectors(uu.T)
                wv = v.new_vectors(nv)
                v.multiply(uv.T, wv)
                wv.copy(v)
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


class AMatrix:

    def __init__(self, a, arch='cpu', copy_data=False):
        self.__arch = arch
        if arch[:3] == 'gpu':
            try:
                from ..algebra import cuda_wrap as cuda
                from ..algebra.dense_cublas import Matrix, Vectors
                self.__op = Matrix(a)
                self.__gpu = cuda
            except:
                if len(arch) > 3 and arch[3] == '!':
                    raise RuntimeError('cannot use GPU')
        else:
            from ..algebra.dense_cpu import Matrix, Vectors
#            from ..algebra.dense_numpy import Matrix, Vectors
            if copy_data:
                self.__op = Matrix(a.copy())
            else:
                self.__op = Matrix(a)
            self.__gpu = None
            #self.__vectors = Vectors(self.__op, shallow=False)
        self.__vectors = Vectors(self.__op, shallow=True)

    def as_operator(self):
        return self.__op

    def as_vectors(self):
        return self.__vectors

    def arch(self):
        return self.__arch

    def gpu(self):
        return self.__gpu

    def dots(self):
        return self.__op.dots()

    def data_type(self):
        return self.__op.data_type()

    def shape(self):
        return self.__op.shape()


class PSVDErrorCalculator:
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
    def set_up(self, op, solver, eigenvectors, shift=False):
        self.op = op.op
        self.solver = solver
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

    def __init__(self, a, err_tol=0, norm='f', max_nsv=0, verb=0):
        self.err_tol = err_tol
        self.norm = norm
        self.max_nsv = max_nsv
        self.verb = verb
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.max_norm = numpy.amax(self.norms)
        self.f_norm = math.sqrt(numpy.sum(self.norms*self.norms))
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
            err_abs = numpy.amax(self.err)
            err_rel = err_abs/self.max_norm
        elif self.norm == 'f':
            self.f -= numpy.sum(sigma*sigma)
            err_abs = math.sqrt(abs(self.f))
            err_rel = err_abs/self.f_norm
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
        if self.err_tol != 0:
            if self.verb > 0:
                print(msg)
            if self.err_tol > 0:
                done = err_rel <= self.err_tol
            else:
                done = err_abs <= abs(self.err_tol)
        else:
            done = (input(msg + ', more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        done = done or self.max_nsv > 0 and self.ncon >= self.max_nsv
        return done


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
        res = solver.convergence_data('residual', i)
        lmd = solver.convergence_data('eigenvalue', i)
        lmd_max = solver.convergence_data('max eigenvalue', i)
        tol = (lmd/lmd_max)**1.5*self.tolerance
        return res >= 0 and res*res <= tol


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a
