# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Truncated SVD demo.

Usage:
  truncated_svd [--help | -h | <data> <rank>]

Arguments:
  data  numpy .npy file containing the matrix.
  rank  the number of singular values and vectors needed

"""

import numpy
import numpy.linalg as nla
from scipy.sparse.linalg import svds
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 2

from raleigh.core.solver import Options
from raleigh.interfaces.truncated_svd import truncated_svd


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


narg = len(sys.argv)
if narg < 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\nUsage:\n')
    print('python truncated_svd.py <data> <rank>')
    exit()
filename = sys.argv[1]
A = numpy.load(filename)
rank = int(sys.argv[2])
arch = 'cpu' if narg < 4 else 'gpu!'

numpy.random.seed(1) # make results reproducible

m = A.shape[0]
if len(A.shape) > 2:
    n = numpy.prod(A.shape[1:])
    A = numpy.reshape(A, (m, n))
else:
    n = A.shape[1]
dtype = A.dtype.type

print('\n--- solving with truncated_svd...\n')
start = time.time()
u, sigma, vt = truncated_svd(A, nsv=rank, arch=arch, verb=1)
stop = time.time()
time_tsvd = stop - start
print('\ntruncated_svd time: %.1e' % time_tsvd)
print('\n%d singular vectors computed' % sigma.shape[0])
D = A - numpy.dot(sigma*u, vt)
err = numpy.amax(_norm(D, axis=1))/numpy.amax(_norm(A, axis=1))
print('\ntruncation error %.1e' % err)
exit()

print('\n--- solving with svds...\n')
start = time.time()
u, sigma, vt = svds(A, k=rank)
stop = time.time()
time_svds = stop - start
print('\nsvds time: %.1e' % time_svds)
print('\n%d singular vectors computed' % sigma.shape[0])
D = A - numpy.dot(sigma*u, vt)
err = numpy.amax(_norm(D, axis=1))/numpy.amax(_norm(A, axis=1))
print('\ntruncation error %.1e' % err)

print('\ndone')
