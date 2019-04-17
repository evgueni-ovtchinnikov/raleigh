# -*- coding: utf-8 -*-
"""
SCR matrix solver tests.

Usage:
    scr_tests [--help | -h | options]

Options:
    -m <mat>, --matrix=<mat>  problem matrix [default: lap3d]
    -n <dim>, --dim=<dim>     mesh sizes nx = ny = nz for lap3d [default: 10]
    -b <blk>, --bsize=<blk>   block CG block size [default: -1]
    -d <dat>, --dtype=<dat>   data type (BLAS prefix s/d) [default: d]
    -s <sft>, --shift=<sft>   shift [default: 0]
    -l <lft>, --left=<lft>    number of eigenvalues left of shift [default: 0]
    -r <rgt>, --right=<rgt>   number of eigenvalues right of shift [default: 0]
    -t <tol>, --tol=<tol>     error/residual tolerance [default: 1e-10]
    -S, --scipy  solve with SciPy eigsh

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

matrix = args['--matrix']
if matrix == 'lap3d':
    nx = int(args['--dim'])
block_size = int(args['--bsize'])
dt = args['--dtype']
sigma = float(args['--shift'])
left = int(args['--left'])
right = int(args['--right'])
tol = float(args['--tol'])
eigsh = args['--scipy']

raleigh_path = '../../..'

import numpy
from scipy.io import mmread
import scipy.sparse as scs
from scipy.sparse.linalg import LinearOperator
import sys
import time

if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

import raleigh.solver
from raleigh.ndarray.cblas_algebra import Vectors
from raleigh.ndarray.mkl import SparseSymmetricMatrix
from raleigh.ndarray.mkl import SparseSymmetricDirectSolver

class MyStoppingCriteria:
    def __init__(self, n):
        self.__n = n
    def satisfied(self, solver):
        if solver.lcon + solver.rcon >= self.__n:
            return True
        return False

def lap3d_matrix(nx, ny, nz, dtype = numpy.float64):
    hx = 1.0
    hy = 1.0
    hz = 1.0
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
else:
    raise ValueError('data type %s not supported' % dt)

if matrix == 'lap3d':
    print('generating %s matrix...' % matrix)
    a, ia, ja = lap3d_matrix(nx, nx, nx, dtype)
else:
    path = 'C:/Users/wps46139/Documents/Data/Matrices/'
    print('reading the matrix from %s...' % matrix)
    M = mmread(path + matrix).tocsr()
    U = scs.triu(M, format = 'csr')
    a = U.data.astype(dtype)
    ia = U.indptr + 1
    ja = U.indices + 1

n = ia.shape[0] - 1
v = Vectors(n, data_type = dtype)
A = SparseSymmetricMatrix(a, ia, ja)
opA = lambda x, y: A.dot(x.data(), y.data())
SSDS = SparseSymmetricDirectSolver(dtype)
opAinv = lambda x, y: SSDS.solve(x.data(), y.data())

print('setting up the solver...')
start = time.time()
SSDS.define_structure(ia, ja)
SSDS.reorder()
SSDS.factorize(a, sigma = sigma)
stop = time.time()
setup_time = stop - start
print('setup time: %.2e' % setup_time)
inertia = SSDS.inertia()
pos = int(inertia[0])
neg = int(inertia[1])
print('positive eigenvalues: %d' % pos)
print('negative eigenvalues: %d' % neg)
left = min(left, neg)
right = min(right, pos)

opt = raleigh.solver.Options()
opt.block_size = block_size
opt.convergence_criteria = raleigh.solver.DefaultConvergenceCriteria()
opt.convergence_criteria.set_error_tolerance('k eigenvector error', tol)
#opt.max_iter = 30
#opt.verbosity = 2
if left < 0 and right < 0:
    opt.stopping_criteria = MyStoppingCriteria(-left)

evp = raleigh.solver.Problem(v, opAinv)
solver = raleigh.solver.Solver(evp)
#solver.set_preconditioner(opAinv)

start = time.time()
solver.solve(v, opt, which = (left, right))
stop = time.time()
solve_time = stop - start
print('after %d iterations, %d converged eigenvalues are:' \
      % (solver.iteration, v.nvec()))
print(numpy.sort(sigma + 1./solver.eigenvalues))
#print(numpy.sort(solver.eigenvalues))
print('solve time: %.2e' % solve_time)

def mv(x):
    y = x.copy()
    SSDS.solve(x, y)
    return y

if eigsh:
    opM = LinearOperator(dtype = dtype, shape = (n, n), \
                         matmat = mv, matvec = mv, rmatvec = mv)
    if left < 0 and right < 0:
        k = -left
    else:
        k = left + right
    print('solving with scipy eigsh...')
    start = time.time()
    vals, vecs = scs.linalg.eigsh(opM, k, tol = 1e-12)
#    vals, vecs = scs.linalg.eigsh(M, k, sigma = sigma, tol = tol)
    stop = time.time()
    eigsh_time = stop - start
    print(numpy.sort(sigma + 1./vals))
    print('eigsh time: %.2e' % eigsh_time)
