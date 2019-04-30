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


def partial_hevp(A, B=None, sigma=0, which=(0, 6), tol=1e-4, verb=0):

    m, n = A.shape
    if m != n:
        raise ValueError('the matrix must be square')
    dtype = A.data.dtype
    solver = SparseSymmetricSolver(dtype=dtype)

    eigenvectors = Vectors(n, data_type=dtype)
    opAinv = lambda x, y: solver.solve(x, y)

    if verb > -1:
        print('setting up the linear system solver...')
    start = time.time()
    solver.analyse(A, sigma)
    solver.factorize()
    neg, pos = solver.inertia()
    stop = time.time()
    setup_time = stop - start
    if verb > -1:
        print('setup time: %.2e' % setup_time)
        print('positive eigenvalues: %d' % pos)
        print('negative eigenvalues: %d' % neg)
    try:
        which[0].upper()
    except:
        left = min(which[0], neg)
        right = min(which[1], pos)
        which = (left, right)

    opt = Options()
    #opt.block_size = block_size
    opt.convergence_criteria = DefaultConvergenceCriteria()
    opt.convergence_criteria.set_error_tolerance('k eigenvector error', tol)
    opt.sigma = sigma
    #opt.max_iter = 30
    opt.verbosity = verb
    evp = Problem(eigenvectors, opAinv)
    evp_solver = Solver(evp)

    start = time.time()
    status = evp_solver.solve(eigenvectors, opt, which=which)
    stop = time.time()
    solve_time = stop - start
    lmd = sigma + 1./evp_solver.eigenvalues
    ind = numpy.argsort(lmd)
    lmd = lmd[ind]
    ne = eigenvectors.nvec()
    if verb > -1:
        print('iterations: %d, solve time: %.2e' % \
              (evp_solver.iteration, solve_time))
    return lmd, eigenvectors.data().T[:,ind], status

