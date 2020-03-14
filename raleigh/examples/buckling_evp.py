# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Computes several eigenvalues and eigenvectors of a buckling problem.

--------------------------------------------------------------------------------
Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on Linux).
--------------------------------------------------------------------------------

Visit https://sparse.tamu.edu/ to download matrices in Matrix Market format
to test on (recommended group: DNVS).

Usage:
    sparse_evp <K> <Ks> <sigma> <nev> [<tol>]

    <K>       name of the Matrix Market file containing stiffness matrix
    <Ks>      name of the Matrix Market file containing stress stiffness matrix
    <alpha>   buckling load shift
    <nev>     number of eigenvalues wanted
    <tol>     error tolerance (default: 1e-6)

NOTE: increasing buckling load shift improves convergence - but increases the
      risk of eigsh's missing lowest load factors.
"""

import numpy
from scipy.io import mmread
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
import scipy.sparse as scs
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 2

from raleigh.interfaces.partial_hevp import partial_hevp


narg = len(sys.argv)
if narg < 4 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\nUsage:\n')
    print('python sparse_evp <K> <Ks> <alpha> <nev> [<tol>]')
    exit()
K = sys.argv[1]
Ks = sys.argv[2]
alpha = float(sys.argv[3])
nev = int(sys.argv[4])
if narg > 5:
    tol = float(sys.argv[5])
else:
    tol = 1e-6

print('\n---reading the matrix K from %s...' % K)
A = mmread(K).tocsr()
print('\n---reading the matrix Ks from %s...' % Ks)
B = mmread(Ks).tocsr()

numpy.random.seed(1) # makes the results reproducible

print('\n---solving with raleigh partial_hevp...')
start = time.time()
vals, vecs, status = partial_hevp(A, B, buckling=True, sigma=-alpha, which=nev, \
                                  tol=tol, verb=0)
stop = time.time()
raleigh_time = stop - start
if status != 0:
    print('partial_hevp execution status: %d' % status)
load_factors = -vals
print('load_factors computed:')
print(load_factors)
print('raleigh time: %.2e' % raleigh_time)

print('\n---solving with scipy eigsh...')
start = time.time()
vals, vecs = scs.linalg.eigsh(A, nev, B, sigma=-alpha, mode='buckling', which='LM', tol=tol)
stop = time.time()
eigsh_time = stop - start
load_factors = -vals[::-1]
print('load_factors computed:')
print(load_factors)
print('eigsh time: %.2e' % eigsh_time)
