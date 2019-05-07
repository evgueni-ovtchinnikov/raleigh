"""
Partial eigenvalue problem solver for a sparse symmetric/Hermitian matrix.

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numpy
import scipy.sparse as scs
import time

from ..ndarray.cblas_algebra import Vectors
from ..ndarray.sparse_algebra import SparseSymmetricMatrix
from ..ndarray.sparse_algebra import SparseSymmetricSolver
from ..solver import Problem, Solver, Options, DefaultConvergenceCriteria


def partial_hevp(A, B=None, T=None, sigma=0, which=6, tol=1e-4, verb=0):

    if B is not None:
        B = SparseSymmetricMatrix(B)
        opB = lambda x, y: B.apply(x, y)
    else:
        opB = None

    if T is None:
        try:
            n = A.size()
            dtype = A.data_type()
            sigma = A.sigma()
            neg, pos = A.inertia()
            solver = A
        except:
            m, n = A.shape
            if m != n:
                raise ValueError('the matrix must be square')
            dtype = A.data.dtype
            solver = SparseSymmetricSolver(dtype=dtype)
            if verb > -1:
                print('setting up the linear system solver...')
            start = time.time()
            solver.analyse(A, sigma, B)
            solver.factorize()
            stop = time.time()
            setup_time = stop - start
            if verb > -1:
                print('setup time: %.2e' % setup_time)
        opAinv = lambda x, y: solver.solve(x, y)
        neg, pos = solver.inertia()
        if verb > -1:
            print('positive eigenvalues: %d' % pos)
            print('negative eigenvalues: %d' % neg)
        try:
            which[0].upper()
        except:
            try:
                left = min(which[0], neg)
                right = min(which[1], pos)
                which = (left, right)
            except:
                which = ('largest', which)
        eigenvectors = Vectors(n, data_type=dtype)
        if B is None:
            evp = Problem(eigenvectors, opAinv)
        else:
            evp = Problem(eigenvectors, opAinv, opB, 'pro')
        evp_solver = Solver(evp)
    else:
        A = SparseSymmetricMatrix(A)
        n = A.size()
        dtype = A.data_type()
        eigenvectors = Vectors(n, data_type=dtype)
        opA = lambda x, y: A.apply(x, y)
        opT = lambda x, y: T.apply(x, y)
        if B is None:
            evp = Problem(eigenvectors, opA)
        else:
            evp = Problem(eigenvectors, opA, opB, 'gen')
        evp_solver = Solver(evp)
        evp_solver.set_preconditioner(opT)
        sigma = None
        which = (which, 0)

    opt = Options()
    #opt.block_size = block_size
    opt.convergence_criteria = DefaultConvergenceCriteria()
    opt.convergence_criteria.set_error_tolerance('k eigenvector error', tol)
    opt.sigma = sigma
    #opt.max_iter = 30
    opt.verbosity = verb

    start = time.time()
    status = evp_solver.solve(eigenvectors, opt, which=which)
    stop = time.time()
    solve_time = stop - start
    if T is None:
        lmd = sigma + 1./evp_solver.eigenvalues
    else:
        lmd = evp_solver.eigenvalues
    ind = numpy.argsort(lmd)
    lmd = lmd[ind]
    ne = eigenvectors.nvec()
    if verb > -1:
        print('iterations: %d, solve time: %.2e' % \
              (evp_solver.iteration, solve_time))
    return lmd, eigenvectors.data().T[:,ind], status

