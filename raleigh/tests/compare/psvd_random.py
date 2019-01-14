# -*- coding: utf-8 -*-
"""Comparisons of partial_svd with scipy svds and svd on smallest singular 
   values of randomly generated matrix.

Usage:
  psvd_random [--help | -h | options] <m> <n>

Arguments:
  m  number of rows
  n  number of rows

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   block CG block size [default: 32]
  -e <err> , --error=<err>   error measure [default: kinematic-eigenvector-err]
  -n <nsv> , --nsv=<nsv>     number of singular values [default: 6]
  -r <alph>, --alpha=<alph>  singular values decay rate [default: 10]
  -s <ths> , --thresh=<ths>  singular values threshold [default: 0.01]
  -t <tol> , --tol=<tol>     error or residual tolerance [default: 1e-3]
  -v <verb>, --verb=<verb>   verbosity level [default: -1]
  -D, --double  use double precision
  -F, --full    compute full SVD too (using scipy.linalg.svd)
  -O, --linop   for svds use LinearOperator instead of ndarray
  -S, --svds    compute truncated SVD using scipy.sparse.linalg.svds

@author: Evgueni Ovtchinnikov, UKRI

Created on Mon Jan 14 10:56:19 2019
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

m = int(args['<m>'])
n = int(args['<n>'])
alpha = float(args['--alpha'])
arch = args['--arch']
block_size = int(args['--bsize'])
nsv = int(args['--nsv'])
th = float(args['--thresh'])
tol = float(args['--tol'])
verb = int(args['--verb'])
err = args['--error']
dble = args['--double']
full = args['--full']
run_svds = args['--svds']
use_op = args['--linop']

import numpy
import numpy.linalg as nla
import scipy.linalg as sla
import sys
import time

from scipy.sparse.linalg import svds, LinearOperator

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)
matrix_path = '../matrix'
if matrix_path not in sys.path:
    sys.path.append(matrix_path)

from random_matrix_for_svd import random_matrix_for_svd
from raleigh.solver import Options
from raleigh.svd import truncated_svd

def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = numpy.linalg.norm(w, axis = 0)
    return s

numpy.random.seed(1) # make results reproducible

if dble:
    dtype = numpy.float64
else:
    dtype = numpy.float32

print('\n--- generating the matrix...')
k = min(m, n)
f_sigma = lambda t: t.astype(dtype)
sigma, u, v, A = random_matrix_for_svd(m, n, k, f_sigma, dtype)
print(sigma[:nsv])
#print(sigma)

opt = Options()
#opt.block_size = block_size
opt.max_iter = 10000
opt.verbosity = verb
opt.convergence_criteria.set_error_tolerance(err, tol)

start = time.time()
sigma_r, u, vt_r = truncated_svd(A, opt, nsv = nsv, largest = False, arch = arch)
stop = time.time()
time_r = stop - start
print('raleigh time: %.1e' % time_r)

n_r = sigma_r.shape[0]
print('\n%d singular vectors computed' % n_r)
print(sigma_r)
n_r = min(n_r, sigma.shape[0])
if n_r > 0:
    err_vec = vec_err(v[:,:n_r], vt_r.transpose()[:,:n_r])
    err_val = abs(sigma_r[:n_r] - sigma[:n_r])
    print('\nmax singular vector error (raleigh): %.1e' % numpy.amax(err_vec))
    print('\nmax singular value error (raleigh): %.1e' % numpy.amax(err_val))

if run_svds:
    start = time.time()
    # fails in single precision for m = n = 1000, nsv = 6, tol = 1e-3/1e-4/0;
    # in double precision, m = n = 2000, nsv = 6, tol = 1e-13 after 20000
    # arpack iterations only 4 singular vectors converged
    u_s, sigma_s, vt_s = svds(A, k = nsv, which = 'SM', tol = tol)
    stop = time.time()
    time_s = stop - start
    ncon = sigma_s.shape[0]
    print('\n%d singular vectors computed in %.1e sec' % (ncon, time_s))
    #print('\n%d singular vectors computed' % sigma_s.shape[0])
    print(sigma_s)
    n_s = min(sigma_s.shape[0], sigma.shape[0])
    if n_s > 0:
        err_vec = vec_err(v[:,:n_s], vt_s.transpose()[:,:n_s])
        err_val = abs(sigma_s[:n_s] - sigma[:n_s])
        print('\nmax singular vector error (svds): %.1e' % numpy.amax(err_vec))
        print('\nmax singular value error (svds): %.1e' % numpy.amax(err_val))

print('\ndone')