# -*- coding: utf-8 -*-
"""Eigenvalue problem for real symmetric SCR matrix.

Usage:
    sparse_evp [--help | -h | options]

Options:
    -a <mat>, --matrix=<mat>  problem matrix [default: lap3d]
    -b <mss>, --mass=<mss>    mass matrix
    -n <dim>, --dim=<dim>     mesh sizes nx = ny = nz for lap3d [default: 10]
    -d <dat>, --dtype=<dat>   data type (BLAS prefix s/d) [default: d]
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
    -I, --invop  first argument of partial_hevp is a SparseSymmetricSolver
    -P, --ilutp  use mkl dcsrilut (incomplete ILU) as a preconditioner

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

from docopt import docopt
import numpy
from scipy.io import mmread
import scipy.sparse as scs
import sys
import time

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.ndarray.sparse_algebra import SparseSymmetricMatrix
from raleigh.ndarray.sparse_algebra import SparseSymmetricSolver
from raleigh.ndarray.sparse_algebra import IncompleteLU
from raleigh.apps.partial_hevp import partial_hevp


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


def lap3d_upper(nx, ny, nz, ax, ay, az):
    return scs.triu(lap3d(nx, ny, nz, ax, ay, az), format='csr')


__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

matrix = args['--matrix']
mass = args['--mass']
if matrix == 'lap3d':
    nx = int(args['--dim'])
dt = args['--dtype']
sigma = float(args['--shift'])
left = int(args['--left'])
right = int(args['--right'])
nev = int(args['--nev'])
tol = float(args['--tol'])
invop = args['--invop']
ilutp = args['--ilutp']

numpy.random.seed(1) # makes the results reproducible

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

if matrix == 'lap3d':
    print('generating %s matrix...' % matrix)
    M = lap3d(nx, nx, nx, 1.0, 1.0, 1.0)
    ia = M.indptr + 1
    n = ia.shape[0] - 1
    if mass is not None:
        B = 2*scs.eye(n, dtype=dtype, format='csr')
    else:
        B = None
else:
    print('reading the matrix from %s...' % matrix)
    M = mmread(matrix).tocsr().astype(dtype)
    n = M.shape[0]
    if mass is not None:
        print('reading the mass matrix from %s...' % mass)
        B = mmread(path + mass).tocsr()
    else:
        B = None

T = None

if invop:
    print('setting up the linear system solver...')
    start = time.time()
    A = SparseSymmetricSolver(dtype=dtype)
    A.analyse(M, sigma, B)
    A.factorize()
    stop = time.time()
    setup_time = stop - start
    print('setup time: %.2e' % setup_time)
else:
    A = M
    if ilutp:
        if dtype != numpy.float64:
            raise ValueError('ILU preconditioner cannot handle data type %s' \
                             % repr(dtype))
        print('setting up the preconditioner...')
        start = time.time()
        T = IncompleteLU(M)
        T.factorize(tol=1e-4, max_fill=10)
        stop = time.time()
        setup_time = stop - start
        print('setup time: %.2e' % setup_time)

if T is not None:
    which = nev
else:
    which = (left, right)

vals, vecs, status = partial_hevp(A, B, T, sigma=sigma, which=which, tol=tol)
nr = vals.shape[0]
if status != 0:
    print('raleighs execution status: %d' % status)
print('converged eigenvalues are:')
print(vals)
