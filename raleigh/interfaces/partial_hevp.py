# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Partial eigenvalue problem solver for a sparse symmetric/Hermitian matrix.

--------------------------------------------------------------------------------
Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on Linux).
--------------------------------------------------------------------------------
"""

import numpy
import time

from ..algebra.dense_cblas import Vectors
from ..algebra.sparse_mkl import SparseSymmetricMatrix
from ..algebra.sparse_mkl import SparseSymmetricSolver
from ..algebra.sparse_mkl import Operator
from ..core.solver import Problem, Solver, Options, DefaultConvergenceCriteria


def partial_hevp(A, B=None, T=None, buckling=False, sigma=0, which=6, tol=1e-4,\
                 verb=0, opt=Options()):
    '''Computes several eigenpairs of sparse real symmetric/Hermitian eigenvalue
    problems using either shift-invert or preconditioning technique.
    Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on
    Linux).

    Parameters
    ----------
    A : either scipy's sparse matrix or raleigh's SparseSymmetricSolver
        In the first case, the (stiffness) matrix of the problem.
        In the second case, the inverse of A - sigma B if B is not None or
        otherwise A - sigma I, where I is the identity.
    B : scipy's sparse matrix
        In buckling case (buckling is True), stress stiffness matrix, otherwise
        mass matrix, which must be positive definite. If None, eigenvalues and
        eigenvectors of A are computed, otherwise those of the generalized
        problem A x = lambda B x.
    T : a Python object
        If T is not None, then A and B must be positive definite.
        Preconditioner (roughly, approximate inverse of A). Must have method
        apply(x, y) that, for a given equally shaped 2D ndarrays x and y with
        the second dimension equal to the problem size, applies preconditioning
        to rows of x and places the results into respective rows of y.
        The method apply(x, y) must act as a self-adjoint positive definite 
        linear operator, i.e. for any x, the matrix numpy.dot(x, y), where y is
        computed by apply(x, y), must be real symmetric/Hermitian and positive
        definite.
    buckling : Boolean
        Flag for buckling mode.
    sigma : float
        Shift inside the spectrum for the sake of faster convergence. Must be
        negative if buckling is True. Ignored if T is not None.
    which : an integer or tuple of two integers
        Specifies which eigenvalues are wanted.
        Integer: if T is not none or buckling is True, then it is
        the number of wanted smallest eigenvalues, otherwise the number of
        wanted eigenvalues nearest to sigma.
        Tuple (k, l): k nearest eigenvalues left from sigma and l nearest
        eigenvalues right from sigma are wanted.
    tol : float
        Eigenvector error tolerance.
    verb : integer
        Verbosity level.
        < 0 : nothing printed
          0 : error and warning messages printed
          1 : + number of iteration and converged eigenvalues printed
          2 : + current eigenvalue iterates, residuals and error estimates
              printed
    opt : an object of class raleigh.solver.Options
        Solver options (see raleigh.solver).

    Returns
    -------
    lmd : one-dimensional numpy array
        Eigenvalues in ascending order.
    x : two-dimensional numpy array
        The matrix of corresponding eigenvectors as columns.
    status : int
        Execution status
        0 : success
        1 : maximal number of iterations exceeded
        2 : no search directions left (bad problem data or preconditioner)
        3 : some of the requested left eigenvalues may not exist
        4 : some of the requested right eigenvalues may not exist
       <0 : error found, error message printed if verb is non-negative
    '''

    if buckling and sigma >= 0:
        raise ValueError('sigma must be negative in buckling mode')

    if B is not None:
        if buckling:
            opB = SparseSymmetricMatrix(A)
        else:
            opB = SparseSymmetricMatrix(B)
    else:
        if buckling:
            raise RuntimeError\
                  ('stress stiffness matrix missing in buckling mode')
        opB = None

    if T is None: # use sparse factorization

        if isinstance(A, SparseSymmetricSolver): # already have factors

            n = A.size()
            dtype = A.data_type()
            sigma = A.sigma()
            neg, pos = A.inertia()
            solver = A

        else: # factorize

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

            # estimate factorization error

            A = SparseSymmetricMatrix(A)
            b = Vectors(n, 3, data_type=dtype)
            x = Vectors(n, 3, data_type=dtype)
            y = Vectors(n, 3, data_type=dtype)
            x.fill_random()
            A.apply(x, b)
            if B is not None:
                opB_ = SparseSymmetricMatrix(B)
            else:
                opB_ = None
            if opB_ is not None:
                opB_.apply(x, y)
                z = y
            else:
                z = x
            s = x.dots(x)
            if sigma != 0:
                b.add(z, -sigma)
            solver.solve(b, y)
            y.add(x, -1)
            if opB_ is not None:
                opB_.apply(y, b)
                z = b
            else:
                z = y
            t = y.dots(y)
            err = numpy.amax(numpy.sqrt(abs(t/s)))
            if err > 0.01:
                msg = 'factorization too inaccurate: relative error > %.1e, ' \
                      + 'consider moving shift slightly'
                if verb > -1:
                    print( msg % err)
                return None, None, -1
            elif verb > -1:
                print('estimated factorization error: %.1e' % err)

            stop = time.time()
            setup_time = stop - start
            if verb > -1:
                print('setup time: %.2e' % setup_time)

        # set up the eigenvalue problem

        opAinv = solver
        neg, pos = solver.inertia()
        if verb > -1:
            print('positive eigenvalues: %d' % pos)
            print('negative eigenvalues: %d' % neg)
        if type(which) is tuple:
            l = len(which)
            if l != 2:
                raise ValueError\
                      ('which must be either integer or tuple of 2 integers')
            left = min(which[0], neg)
            right = min(which[1], pos)
            which = (left, right)
        else:
            if buckling:
                if which < neg:
                    which = (neg, 0)
                else:
                    which = (neg, which - neg)
##            else:
##                if neg < 1:
##                    which = (0, which)
##                elif pos < 1:
##                    which = (which, 0)
        eigenvectors = Vectors(n, data_type=dtype)
        if B is None:
            evp = Problem(eigenvectors, opAinv)
        else:
            evp = Problem(eigenvectors, opAinv, opB, 'pro')
        evp_solver = Solver(evp)

    else: # use preconditioning

        if buckling:
            msg = 'preconditioning for buckling problem not supported'
            raise ValueError(msg)

        A = SparseSymmetricMatrix(A)
        n = A.size()
        dtype = A.data_type()
        eigenvectors = Vectors(n, data_type=dtype)
        opA = A
        opT = Operator(T)
        if B is None:
            evp = Problem(eigenvectors, opA)
        else:
            evp = Problem(eigenvectors, opA, opB, 'gen')
        evp_solver = Solver(evp)
        evp_solver.set_preconditioner(opT)
        sigma = None
        if type(which) is tuple:
            raise ValueError\
                  ('which must be integer if preconditioning is used')
        which = (which, 0)

    # solve the eigenvalue problem

    opt.convergence_criteria = DefaultConvergenceCriteria()
    opt.convergence_criteria.set_error_tolerance('k eigenvector error', tol)
    opt.sigma = sigma

    start = time.time()
    status = evp_solver.solve(eigenvectors, opt, which=which)
    if status < 0:
        return None, None, status
    stop = time.time()
    solve_time = stop - start
    if T is None:
        if buckling:
            lmd = sigma/(1 - 1/evp_solver.eigenvalues)
        else:
            lmd = sigma + 1./evp_solver.eigenvalues
    else:
        lmd = evp_solver.eigenvalues
    if buckling:
        ind = numpy.argsort(-lmd)
    else:
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
