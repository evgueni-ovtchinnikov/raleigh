# -*- coding: utf-8 -*-
"""
Simple solver tests.

Usage:
    simple_tests [--help | -h | options]

Options:
    -p <pro>, --problem=<pro>  eigenvalue problem type [default: std]
    -m <mat>, --matrix=<mat>   problem matrix [default: diag]
    -n <dim>, --dim=<dim>      problem size [default: 100]
    -d <dat>, --dtype=<dat>    data type (BLAS prefix s/d/c/z) [default: d]
    -l <lft>, --left=<lft>     number of left eigenvalues wanted [default: 6]
    -r <lft>, --right=<lft>    number of right eigenvalues wanted [default: 0]
    -t <tol>, --vtol=<tol>     eigenvector error tolerance [default: 1e-8]
    -b <blk>, --bsize=<blk>    block CG block size [default: -1]
    -v <vrb>, --verb=<vrb>     verbosity [default: 0]
    -P, --precond  with preconditioning

Created on Thu Aug  2 10:57:27 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numpy
import sys

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)
import raleigh.solver
from raleigh.algebra.dense_cpu import Vectors, Matrix

try:
    from docopt import docopt
    have_docopt = True
except:
    have_docopt = False

def test():
    '''
    >>> test()
    ... # doctest: +NORMALIZE_WHITESPACE
    after 58 iterations, 6 converged eigenvalues are:
    [ 1. 2. 3. 4. 5. 6.]
    '''

    if have_docopt:
        __version__ = '0.1.0'
        args = docopt(__doc__, version=__version__)
        problem = args['--problem']
        matrix = args['--matrix']
        n = int(args['--dim'])
        dt = args['--dtype']
        left = int(args['--left'])
        right = int(args['--right'])
        vec_tol = float(args['--vtol'])
        block_size = int(args['--bsize'])
        verbosity = int(args['--verb'])
        with_prec = args['--precond']
    else:
        problem = 'std'
        matrix = 'diag'
        n = 100
        dt = 'd'
        left = 6
        right = 0
        vec_tol = 1e-8
        block_size = -1
        verbosity = 0
        with_prec = False

    numpy.random.seed(1) # to debug - makes the results reproducible

    if dt == 's':
        dtype = numpy.float32
    elif dt == 'd':
        dtype = numpy.float64
    elif dt == 'c':
        dtype = numpy.complex64
    elif dt == 'z':
        dtype = numpy.complex128
    else:
        raise ValueError('data type %s not supported' % dt)

    opt = raleigh.solver.Options()
    opt.block_size = block_size
    opt.convergence_criteria = raleigh.solver.DefaultConvergenceCriteria()
    opt.convergence_criteria.set_error_tolerance('eigenvector error', vec_tol)
    opt.verbosity = verbosity

    v = Vectors(n, data_type=dtype)

    if matrix[0] == 'c':
        if dt == 's' or dt == 'd':
            raise ValueError('central differences matrix requires complex data')
        d = 1j*numpy.ones((n - 1,), dtype=dtype)
        A = numpy.diag(d, 1) - numpy.diag(d, -1)
        operatorA = Matrix(A)
    else:
        a = numpy.asarray([i + 1 for i in range(n)]).astype(dtype)
        operatorA = Matrix(numpy.diag(a))
    opA = lambda x, y: operatorA.apply(x, y)

    if problem[0] != 's':
        b = 2*numpy.ones((n,), dtype=dtype)
        operatorB = Matrix(numpy.diag(b))
        opB = lambda x, y: operatorB.apply(x, y)
    else:
        opB = None

    if problem[0] == 'p':
        evp = raleigh.solver.Problem(v, opA, opB, 'pro')
    else:
        evp = raleigh.solver.Problem(v, opA, opB)
    solver = raleigh.solver.Solver(evp)

    if with_prec:
        if problem[0] == 'p':
            raise ValueError('preconditioning does not work for matrix product')
        operatorP = Matrix(numpy.diag(1/a))
        opP = lambda x, y: operatorP.apply(x, y)
        solver.set_preconditioner(opP)

    solver.solve(v, opt, which=(left, right))
    print('after %d iterations, %d converged eigenvalues are:' \
          % (solver.iteration, v.nvec()))
    print(solver.eigenvalues)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
