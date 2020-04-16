# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Computes several eigenvalues and eigenvectors of a real symmetric matrix.

--------------------------------------------------------------------------------
Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on Linux).
Latest MKL can be installed by pip install --user mkl
--------------------------------------------------------------------------------

Visit https://sparse.tamu.edu/ to download matrices in Matrix Market format
to test on (recommended group: DNVS).

Usage:
    sparse_evp <matrix> <nev> [<sigma>, [<tol>]]

    <matrix>  the name of the Matrix Market file containing problem matrix or
              parameters of 3D discretized Laplacian (mesh sizes nx, ny, nz
              and domain sizes ax, ay, az)
    <sigma>   shift (default: 0)
    <nev>     number of eigenvalues wanted nearest to the shift
    <tol>     error tolerance (default: 1e-6)
"""

import numpy
from scipy.io import mmread
from scipy.sparse.linalg import eigsh
import scipy.sparse as scs
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 2

try:
    from raleigh.interfaces.partial_hevp import partial_hevp
except:
    print('This script requires MKL 10.3 or later, sorry!')
    exit()


narg = len(sys.argv)
if narg < 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\nUsage:\n')
    print('python sparse_evp <matrix> <nev> [<sigma>, [<tol>]]')
    exit()
matrix = sys.argv[1]
nev = int(sys.argv[2])
if narg > 3:
    sigma = float(sys.argv[3])
else:
    sigma = 0.0
if narg > 4:
    tol = float(sys.argv[4])
else:
    tol = 1e-6

i = matrix.find('.mtx')
if i < 0: # the file contains 3D Laplacian parameters
    f = open(matrix, 'r')
    data = f.read()
    par = data.split()
    nx = int(par[0])
    ny = int(par[1])
    nz = int(par[2])
    ax = float(par[3])
    ay = float(par[4])
    az = float(par[5])
    from raleigh.examples.laplace import lap3d
    print('\n---generating discretized 3D Laplacian matrix...')
    A = lap3d(nx, ny, nz, ax, ay, az)
else: # the file contains the matrix in Matrix Market format
    print('\n---reading the matrix from %s...' % matrix)
    A = mmread(matrix).tocsr()

numpy.random.seed(1) # makes the results reproducible

print('\n---solving with raleigh partial_hevp...')
start = time.time()
vals, vecs, status = partial_hevp(A, sigma=sigma, which=nev, tol=tol, verb=-1)
stop = time.time()
raleigh_time = stop - start
if status != 0:
    print('partial_hevp execution status: %d' % status)
print('converged eigenvalues are:')
print(vals)
print('raleigh time: %.2e' % raleigh_time)

print('\n---solving with scipy eigsh...')
start = time.time()
vals, vecs = scs.linalg.eigsh(A, nev, sigma=sigma, which='LM', tol=tol)
stop = time.time()
eigsh_time = stop - start
print(vals)
print('eigsh time: %.2e' % eigsh_time)
