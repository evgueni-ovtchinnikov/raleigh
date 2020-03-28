# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Computes several leftmost eigenvalues alpha and corresponding eigenvectors v
of the buckling problem

    (K + alpha Ks)v = 0,

where K is the structure stiffness matrix, Ks is the stress stiffness matrix,
alpha is the buckling load factor and v is the buckling mode shape. Both K and
Ks are real symmetric, K is positive definite and Ks is generally indefinite.

The problem at hand is solved by raleigh's partial_hevp in buckling mode and,
for the sake of performance comparison, by scipy's eigsh in buckling mode.

In both solvers, the problem is transformed into

    (K + alpha0 Ks)v = beta K v,

for some positive alpha0 near alpha's of interest, and solved by inverse
iterations.

Mind that while placing alpha0 near the buckling load factors of interest
improves performance of both partial_svd anf eigsh, a value of alpha0
that is larger than the lowest buckling load factor may cause eigsh
to miss few lowest load factors (e.g. try one of the pairs of matrices
suggested below with alpha0 = 2.5 or higher).

A set of test matrices K and Ks of various sizes (from 8902 to 577643) in
Matrix Market format can be obtained from

https://www.dropbox.com/sh/vuloiyb3tjk3c33/AAAiyAcP6_dh7oC1UFdUiRp9a?dl=0

(soon to be also available on https://sparse.tamu.edu/).

--------------------------------------------------------------------------------
Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on Linux).
The latest MKL can be istalled by pip install --user mkl.
--------------------------------------------------------------------------------

Usage:
    buckling_evp <path> <K> <Ks> <alpha0> <nev> [<tol>]

    <path>    data folder path
    <K>       name of the Matrix Market file containing stiffness matrix
    <Ks>      name of the Matrix Market file containing stress stiffness matrix
    <alpha0>  buckling factor shift: a positive value near the leftmost buckling
              load factors of interest (see NOTE below)
    <nev>     number of buckling modes wanted
    <tol>     error tolerance (optional, default: 1e-6)
"""

import numpy
from scipy.io import mmread
from scipy.sparse.linalg import eigsh
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 2

from raleigh.interfaces.partial_hevp import partial_hevp


narg = len(sys.argv)
if narg < 6 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\nUsage:\n')
    print('python buckling_evp <path> <K> <Ks> <alpha0> <nev> [<tol>]')
    exit()
path = sys.argv[1]
if not path.endswith(('/', '\\')):
    path += '/'
K = path + sys.argv[2]
Ks = path + sys.argv[3]
alpha0 = float(sys.argv[4])
nev = int(sys.argv[5])
if narg > 6:
    tol = float(sys.argv[6])
else:
    tol = 1e-6

print('\n---reading the matrix K from %s...' % K)
A = mmread(K).tocsr()
print('\n---reading the matrix Ks from %s...' % Ks)
B = mmread(Ks).tocsr()

numpy.random.seed(1) # makes the results reproducible

print('\n---solving with raleigh partial_hevp...')
start = time.time()
vals, vecs, status = partial_hevp(A, B, buckling=True, sigma=-alpha0, \
                                  which=nev, tol=tol, verb=0)
stop = time.time()
raleigh_time = stop - start
if status != 0:
    print('partial_hevp execution status: %d' % status)
load_factors = -vals
print('buckling load factors computed:')
print(load_factors)
print('partial_hevp time: %.2e' % raleigh_time)

print('\n---solving with scipy eigsh...')
start = time.time()
vals, vecs = eigsh(A, nev, B, sigma=-alpha0, mode='buckling', which='LM', \
                   tol=tol)
stop = time.time()
eigsh_time = stop - start
load_factors = -vals[::-1]
print('buckling load factors computed:')
print(load_factors)
print('eigsh time: %.2e' % eigsh_time)
