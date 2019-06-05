# -*- coding: utf-8 -*-
"""Computes several eigenvalues and eigenvectors of a real symmetric matrix.

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
    -e <nev>, --nev=<nev>     number of eigenvalues wanted: largest if left and
                              right are set to 0 in shift-invert mode
                              or smallest if preconditioning is used
                              [default: 6]
    -t <tol>, --tol=<tol>     error/residual tolerance [default: 1e-10]
    -v <vrb>, --verb=<vrb>    verbosity level (negative suppresses all output)
                              [default: 0]
    -I, --invop  first argument of partial_hevp is a SparseSymmetricSolver
    -P, --ilutp  use mkl dcsrilut (incomplete ILU) as a preconditioner

@author: Evgueni Ovtchinnikov, UKRI-STFC
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

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

##from raleigh.ndarray.sparse_algebra import SparseSymmetricMatrix
##from raleigh.ndarray.sparse_algebra import SparseSymmetricSolver
##from raleigh.ndarray.sparse_algebra import IncompleteLU
from raleigh.algebra.sparse_mkl import SparseSymmetricSolver
from raleigh.algebra.sparse_mkl import IncompleteLU
from raleigh.drivers.partial_hevp import partial_hevp


def lap1d(n, a):
    h = a/(n + 1)
    d = numpy.ones((n,))/(h*h)
    return scs.spdiags([-d, 2*d, -d], [-1, 0, 1], n, n, format='csr')


def lap2d(nx, ny, ax, ay):
    L = lap1d(nx, ax)
    Ly = lap1d(ny, ay)
    L = scs.csr_matrix(scs.kron(scs.eye(ny), L) + scs.kron(Ly, scs.eye(nx)))
    return L


def lap3d(nx, ny, nz, ax, ay, az):
    L = lap2d(nx, ny, ax, ay)
    Lz = lap1d(nz, az)
    L = scs.csr_matrix(scs.kron(scs.eye(nz), L) + scs.kron(Lz, scs.eye(nx*ny)))
    return L


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
else:
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

numpy.random.seed(1) # makes the results reproducible

if matrix == 'lap3d':
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
    M = mmread(matrix).tocsr().astype(dtype)
    n = M.shape[0]
    if mass is not None:
        if verb > -1:
            print('reading the mass matrix from %s...' % mass)
        B = mmread(path + mass).tocsr()
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
nr = vals.shape[0]
if status != 0 and verb > -1:
    print('partial_hevp execution status: %d' % status)
if verb > -1:
    print('converged eigenvalues are:')
    print(vals)
