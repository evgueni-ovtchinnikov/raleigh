# -*- coding: utf-8 -*-
"""
SCR matrix solver tests.

Usage:
    scr_tests [--help | -h | options]

Options:
    -a <mat>, --matrix=<mat>  problem matrix [default: lap3d]
    -b <mss>, --mass=<mss>    mass matrix
    -n <dim>, --dim=<dim>     mesh sizes nx = ny = nz for lap3d [default: 10]
    -d <dat>, --dtype=<dat>   data type (BLAS prefix s/d) [default: d]
    -s <sft>, --shift=<sft>   shift [default: 0]
    -l <lft>, --left=<lft>    number of eigenvalues left of shift [default: 0]
    -r <rgt>, --right=<rgt>   number of eigenvalues right of shift [default: 0]
    -t <tol>, --tol=<tol>     error/residual tolerance [default: 1e-10]
    -I, --invop  first argument of partial_hevp is a SparseSymmetricSolver
    -S, --scipy  solve also with SciPy eigsh

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

matrix = args['--matrix']
mass = args['--mass']
if matrix == 'lap3d':
    nx = int(args['--dim'])
dt = args['--dtype']
sigma = float(args['--shift'])
left = int(args['--left'])
right = int(args['--right'])
tol = float(args['--tol'])
eigsh = args['--scipy']
invop = args['--invop']

import numpy
from scipy.io import mmread
import scipy.sparse as scs
from scipy.sparse.linalg import LinearOperator
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.ndarray.sparse_algebra import SparseSymmetricMatrix
from raleigh.ndarray.sparse_algebra import SparseSymmetricSolver
from raleigh.apps.partial_hevp import partial_hevp as raleighs

def lap3d_matrix(nx, ny, nz, dtype = numpy.float64):
    hx = 1.0/(nx + 1)
    hy = 1.0/(ny + 1)
    hz = 1.0/(nz + 1)
    xh = 1.0/(hx*hx)
    yh = 1.0/(hy*hy)
    zh = 1.0/(hz*hz)
    n = nx*ny*nz
    ia = numpy.ndarray((n + 1,), dtype = numpy.int32)
    ia[:] = 0
    i = 0
    j = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                j += 1
                if x < nx - 1:
                    j += 1
                if y < ny - 1:
                    j += 1
                if z < nz - 1:
                    j += 1
                i += 1
                ia[i] = j
    nnz = ia[n]
    a = numpy.ndarray((nnz,), dtype = dtype)
    ja = numpy.ndarray((nnz,), dtype = numpy.int32)
    i = 0
    j = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                a[j] = 2*(xh + yh + zh)
                ja[j] = i + 1
                j += 1
                if x < nx - 1:
                    a[j] = -xh
                    ja[j] = i + 2
                    j += 1
                if y < ny - 1:
                    a[j] = -yh
                    ja[j] = i + nx + 1
                    j += 1
                if z < nz - 1:
                    a[j] = -zh
                    ja[j] = i + nx*ny + 1
                    j += 1
                i += 1
    ia += 1
    return a, ia, ja

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

if matrix == 'lap3d':
    print('generating %s matrix...' % matrix)
    a, ia, ja = lap3d_matrix(nx, nx, nx, dtype)
    n = ia.shape[0] - 1
    M = scs.csr_matrix((a, ja - 1, ia - 1), shape = (n, n))
    if mass is not None:
        B = 2*scs.eye(n, dtype=dtype, format='csr')
    else:
        B = None
else:
    path = 'C:/Users/wps46139/Documents/Data/Matrices/'
    print('reading the matrix from %s...' % matrix)
    M = mmread(path + matrix).tocsr().astype(dtype)
    if mass is not None:
        print('reading the mass matrix from %s...' % mass)
        B = mmread(path + mass).tocsr()
    else:
        B = None

if invop:
    A = SparseSymmetricSolver(dtype=dtype)
    print('setting up the linear system solver...')
    start = time.time()
    A.analyse(M, sigma, B)
    A.factorize()
    stop = time.time()
    setup_time = stop - start
    print('setup time: %.2e' % setup_time)
    T = None
else:
    A = M
    I = scs.eye(n, dtype=dtype)
    T = SparseSymmetricMatrix(I)

if T is not None:
    which = left
elif left < 0 and right < 0:
    which = -right
else:
    which = (left, right)

vals_r, vecs_r, status = raleighs(A, B, T, sigma=sigma, which=which, tol=tol)
nr = vals_r.shape[0]
if status != 0:
    print('raleighs execution status: %d' % status)
print('converged eigenvalues are:')
print(vals_r)

if not eigsh: exit()

if not invop:
    A = SparseSymmetricSolver(dtype=dtype)
    print('setting up the linear system solver...')
    start = time.time()
    A.analyse(M, sigma, B)
    A.factorize()
    stop = time.time()
    setup_time = stop - start
    print('setup time: %.2e' % setup_time)
solver = A.solver()

def mv(x):
    y = x.copy()
    solver.solve(x, y)
    return y

def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = numpy.linalg.norm(w, axis = 0)
    return s

opM = LinearOperator(dtype=dtype, shape=(n, n), \
                     matmat=mv, matvec=mv, rmatvec=mv)
if left < 0 and right < 0:
    k = -right
else:
    k = left + right
print('solving with scipy eigsh...')
start = time.time()
# which = 'BE' did not converge in reasonable time for k = 5:
vals, vecs = scs.linalg.eigsh(opM, k, which='LM', tol=1e-12)
# far too slow:
#    vals, vecs = scs.linalg.eigsh(M, k, sigma=sigma, which='BE', tol=tol)
stop = time.time()
eigsh_time = stop - start
print(numpy.sort(sigma + 1./vals))
print('eigsh time: %.2e' % eigsh_time)
ns = vals.shape[0]
print(vec_err(vecs[:, :ns], vecs_r[:, :nr]))
