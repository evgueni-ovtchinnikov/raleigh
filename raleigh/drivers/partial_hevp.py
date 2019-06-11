# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)
# This software is distributed under a BSD licence, see ../LICENSE.txt.
"""
Partial eigenvalue problem solver for a sparse symmetric/Hermitian matrix.
"""

import numpy
import scipy.sparse as scs
import time

from ..algebra.dense_cblas import Vectors
from ..algebra.sparse_mkl import SparseSymmetricMatrix
from ..algebra.sparse_mkl import SparseSymmetricSolver
from ..algebra.sparse_mkl import Operator
from ..core.solver import Problem, Solver, Options, DefaultConvergenceCriteria


def partial_hevp(A, B=None, T=None, sigma=0, which=6, tol=1e-4, verb=0):
    '''Computes several eigenpairs of sparse real symmetric/Hermitian eigenvalue
    problems using either shift-invert or preconditioning technique.
    Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on
    Linux).

    Parameters
    ----------
    A : either scipy's sparse matrix or raleigh's SparseSymmetricSolver
        In the first case, the (stiffness) matrix of the problem.
        In the second case, the inverse of A - sigma B if B is not None or
        A - sigma I, where I is the identity, otherwise.
    B : scipy's sparse matrix
        Mass matrix. If None, eigenvalues and eigenvectors of A are computed,
        otherwise those of the generalized problem A x = lambda B x.
    T : a Python object
        If T is not None, then A must be positive definite.
        Preconditioner (roughly, approximate inverse of A). Must have method
        apply(x, y) that, for a given equally shaped 2D ndarrays x and y with
        the second dimension equal to the problem size, applies preconditioning
        to rows of x and places the results into respective rows of y.
        The method apply(x, y) must act as a self-adjoint positive definite 
        linear operator, i.e. for any x, the matrix numpy.dot(x, y), where y is
        computed by apply(x, y), must be real symmetric/Hermitian and positive
        definite.
    sigma : float
        Ignored if T is not None. Otherwise specifies value in the vicinity of
        which the wanted eigenvalues are situated.
    which : an integer or tuple of integers
        Specifies which eigenvalues are wanted. If T is not none, then it is
        the number of smallest eigenvalues. Otherwise, if it is an integer k,
        then k eigenvalues nearest to sigma will be computed, and if it is a
        tuple (k, l), then k neares eigenvalues left from sigma and l nearest
        eigenvalues right from sigma will be computed.
    tol : float
        Eigenvector error tolerance.
    verb : integer
        Verbosity level.
        < 0 : nothing printed
          0 : error and warning messages printed
          1 : + number of iteration and converged eigenvalues printed
          2 : + current eigenvalue iterates, residuals and error estimates
              printed
    '''

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
            l = len(which)
            if l != 2:
                raise ValueError\
                      ('which must be either integer or tuple of 2 integers')
            left = min(which[0], neg)
            right = min(which[1], pos)
            which = (left, right)
        except:
            pass
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
        op = Operator(T)
        opT = lambda x, y: op.apply(x, y)
        if B is None:
            evp = Problem(eigenvectors, opA)
        else:
            evp = Problem(eigenvectors, opA, opB, 'gen')
        evp_solver = Solver(evp)
        evp_solver.set_preconditioner(opT)
        sigma = None
        which = (which, 0)

    opt = Options()
    opt.convergence_criteria = DefaultConvergenceCriteria()
    opt.convergence_criteria.set_error_tolerance('k eigenvector error', tol)
    opt.sigma = sigma
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
    x = eigenvectors.data().T
    if ne > 0:
        x = x[:, ind]
    return lmd, x, status
