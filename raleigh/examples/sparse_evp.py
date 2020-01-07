# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Computes several eigenvalues and eigenvectors of a real symmetric matrix.

--------------------------------------------------------------------------------
Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on Linux).
--------------------------------------------------------------------------------

Visit https://sparse.tamu.edu/ to download matrices (in Matrix Market format)
to test on. Recommended group: DNVS. Best performance compared to scipy eigsh
is achieved on matrices with large clusters such as DNVS/shipsec1 and in
low-to-moderate accuracy in eigenvectors (1e-3 to 1e-6, corresponds to single
to double machine precision in eigenvalues).

Usage:
    sparse_evp [--help | -h | options]

Options:
    -a <mat>, --matrix=<mat>  problem matrix (name of a Matrix Market file or
                              lap3d) [default: lap3d]
    -b <mss>, --mass=<mss>    mass matrix if problem is generalized
    -n <dim>, --dim=<dim>     mesh sizes nx = ny = nz for lap3d [default: 10]
    -s <sft>, --shift=<sft>   shift (ignored in preconditioned mode) [default: 0]
    -l <lft>, --left=<lft>    number of eigenvalues left of shift (shift-invert
                              mode) [default: 0]
    -r <rgt>, --right=<rgt>   number of eigenvalues right of shift (shift-ivert
                              mode) [default: 6]
    -e <nev>, --nev=<nev>     number of eigenvalues wanted: nearest to the shift
                              in shift-invert mode if left and right are zeros
                              or smallest if preconditioning is used
                              [default: 6]
    -t <tol>, --tol=<tol>     error/residual tolerance [default: 1e-6]
    -v <vrb>, --verb=<vrb>    verbosity level (negative suppresses all output)
                              [default: 0]
    -I, --invop  first argument of partial_hevp is a SparseSymmetricSolver
    -P, --ilutp  use mkl dcsrilut (incomplete ILU) as a preconditioner
    -C, --check  check the eigenvector errors by solving twice
"""

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import numpy
from scipy.io import mmread
import scipy.sparse as scs
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 1

try:
    from raleigh.algebra.sparse_mkl import SparseSymmetricSolver
    from raleigh.algebra.sparse_mkl import IncompleteLU
    from raleigh.interfaces.partial_hevp import partial_hevp
except:
    print('This script requires MKL 10.3 or later, sorry!')
    exit()


def norm(a, axis):
    return numpy.apply_along_axis(numpy.linalg.norm, axis, a)


def vec_err(u, v, B=None):
    if B is None:
        w = v.copy()
    else:
        w = B.dot(v)
    q = numpy.dot(u.T, w)
    w = numpy.dot(u, q) - v
    s = norm(w, axis = 0)
    return s


if have_docopt:
    args = docopt(__doc__, version=__version__)
    matrix = args['--matrix']
    mass = args['--mass']
    if matrix == 'lap3d':
        nx = int(args['--dim'])
    sigma = float(args['--shift'])
    left = int(args['--left'])
    right = int(args['--right'])
    nev = int(args['--nev'])
    tol = float(args['--tol'])
    verb = int(args['--verb'])
    invop = args['--invop']
    ilutp = args['--ilutp']
    check = args['--check']
else:
    print('\n=== docopt not found, using default options...\n')
    matrix = 'lap3d'
    nx = 10
    mass = None
    sigma = 0
    left = 0
    right = 6
    nev = 6
    tol = 1e-10
    verb = 0
    invop = False
    ilutp = False
    check = False

numpy.random.seed(1) # makes the results reproducible

if matrix == 'lap3d':
    from raleigh.examples.laplace import lap3d
    if verb > -1:
        print('generating discretized 3D Laplacian matrix...')
    M = lap3d(nx, nx, nx, 1.0, 1.0, 1.0)
    ia = M.indptr + 1
    n = ia.shape[0] - 1
    if mass is not None: # just a simple generalized problem test
        B = 2*scs.eye(n, format='csr')
    else:
        B = None
else:
    if verb > -1:
        print('reading the matrix from %s...' % matrix)
    M = mmread(matrix).tocsr()
    n = M.shape[0]
    if mass is not None:
        if verb > -1:
            print('reading the mass matrix from %s...' % mass)
        B = mmread(mass).tocsr()
    else:
        B = None

T = None

if invop:
    if verb > -1:
        print('setting up the linear system solver...')
    start = time.time()
    A = SparseSymmetricSolver()
    A.analyse(M, sigma, B)
    A.factorize()
    stop = time.time()
    setup_time = stop - start
    if verb > -1:
        print('setup time: %.2e' % setup_time)
else:
    A = M
    if ilutp:
        if verb > -1:
            print('setting up the preconditioner...')
        start = time.time()
        T = IncompleteLU(M)
        T.factorize()
        stop = time.time()
        setup_time = stop - start
        if verb > -1:
            print('setup time: %.2e' % setup_time)

if T is not None or left == 0 and right == 0:
    which = nev
else:
    which = (left, right)

vals, vecs, status = partial_hevp(A, B, T, sigma=sigma, which=which, tol=tol, \
                                  verb=verb)
if status != 0 and verb > -1:
    print('partial_hevp execution status: %d' % status)
if verb > -1:
    print('converged eigenvalues are:')
    print(vals)

if check:
    '''Estimate eigenvector errors by solving twice.

    Rationale:
    Consider for simplicity the case of one eigenvector computed.
    Let u1 and u2 be the two approximations to a given eigenvector u computed
    by two calls to partial_evp. Since the initial eigenvector guesses are
    random, u1 and u2 are different, and errors (I - P)u1 and (I - P)u2,
    where P is the orthogonal projector onto u and I is the identity, are
    essentially random vectors, and hence nearly orthogonal. Hence, the norm
    of u2 - u1 cannot be significantly less than the largest of the two errors
    and, at the same time, cannot be significantly greater than this error
    multiplied by the square root of 2, i.e. is very close to the actual error.
    '''
    nev = vals.shape[0]
    lft = numpy.sum(vals < sigma)
    rgt = nev - lft
    vls, vcs, status = partial_hevp(A, B, T, sigma=sigma, which=which, tol=tol, \
                                    verb=verb)
    if status != 0 and verb > -1:
        print('partial_hevp execution status: %d' % status)
    ne = vls.shape[0]
    lt = numpy.sum(vls < sigma)
    rt = ne - lt
    left = min(lft, lt)
    right = min(rgt, rt)
    print('eigenvector errors:')
    if left > 0:
        errs = vec_err(vecs[:, lft - left : lft], vcs[:, lt - left : lt], B)
        print(errs)
    if right > 0:
        errs = vec_err(vecs[:, lft : lft + right], vcs[:, lt : lt + right], B)
        print(errs)
