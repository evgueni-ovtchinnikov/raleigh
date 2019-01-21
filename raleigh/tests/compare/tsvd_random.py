# -*- coding: utf-8 -*-
"""Comparisons of truncated_svd with scipy svds and svd on randomly
   generated matrix.

Usage:
  tsvd_random [--help | -h | options] <m> <n> <k>

Arguments:
  m  number of rows
  n  number of rows
  k  number of non-zero singular values

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   block CG block size [default: -1]
  -e <err> , --error=<err>   error measure [default: kinematic-eigenvector-err]
  -r <alph>, --alpha=<alph>  singular values decay rate [default: 100]
  -s <ths> , --thresh=<ths>  singular values threshold [default: 0.01]
  -t <tol> , --tol=<tol>     error or residual tolerance [default: 1e-3]
  -v <verb>, --verb=<verb>   verbosity level [default: -1]
  -d, --double  use double precision
  -f, --full    compute full SVD too (using scipy.linalg.svd)
  -o, --linop   for svds use LinearOperator instead of ndarray
  -p, --ptb     add random perturbation to make the matrix full rank
  -z, --shift   shift to zero average

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

m = int(args['<m>'])
n = int(args['<n>'])
k = int(args['<k>'])
alpha = float(args['--alpha'])
arch = args['--arch']
block_size = int(args['--bsize'])
th = float(args['--thresh'])
tol = float(args['--tol'])
verb = int(args['--verb'])
err = args['--error']
dble = args['--double']
full = args['--full']
use_op = args['--linop']
ptb = args['--ptb']
shift = args['--shift']

import numpy
import numpy.linalg as nla
import scipy.linalg as sla
from sklearn.decomposition import TruncatedSVD #, PCA
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

class MyStoppingCriteria:
    def __init__(self, a):
        self.th = 1.0
        self.m = -1
        self.iteration = -1
    def set_threshold(self, th):
        self.th = th*th
    def set_how_many(self, m):
        self.m = m
    def satisfied(self, solver):
        self.iteration = solver.iteration
        if solver.rcon < 1:
            return False
        vmin = max(0, numpy.amin(solver.eigenvalues))
        done = vmin < self.th or self.m > 0 and solver.rcon >= self.m
        return done

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
f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, dtype)
if ptb:
    a = 2*numpy.random.rand(m, n).astype(dtype) - 1
    s = numpy.linalg.norm(a, axis = 0)
    A += a*(sigma0[-1]/s)
#    A += a*(min(sigma0[-1], 1e-3)/s)

if block_size < 1:
    b = max(1, min(m, n)//100)
    block_size = 32
    while block_size <= b - 16:
        block_size += 32
    print('using block size %d' % block_size)

print('\n--- solving with raleigh.svd.truncated_svd...')

# set raleigh solver options
opt = Options()
if th > 0:
    opt.block_size = block_size
opt.max_iter = 1000
opt.verbosity = verb
opt.convergence_criteria.set_error_tolerance(err, tol)
opt.stopping_criteria = MyStoppingCriteria(A)
if th > 0:
    opt.stopping_criteria.set_threshold(th)
else:
    opt.stopping_criteria.set_threshold(0)

start = time.time()
if th > 0:
    sigma_r, u, vt_r = truncated_svd(A, opt, arch = arch, shift = shift)
else:
    sigma_r, u, vt_r = truncated_svd(A, opt, nsv = block_size, arch = arch, \
                                 shift = shift)
stop = time.time()
time_r = stop - start
iter_r = opt.stopping_criteria.iteration
print('raleigh time: %.1e' % time_r)

n_r = sigma_r.shape[0]
print('\n%d singular vectors computed in %d iterations' % (n_r, iter_r))
if not ptb and n_r > 0:
    n_r = min(n_r, sigma0.shape[0])
    err_vec = vec_err(v0[:,:n_r], vt_r.transpose()[:,:n_r])
    err_val = abs(sigma_r[:n_r] - sigma0[:n_r])
    print('\nmax singular vector error (raleigh): %.1e' % numpy.amax(err_vec))
    print('\nmax singular value error (raleigh): %.1e' % numpy.amax(err_val))
#else:
#    #print(sigma)
#    print(sigma_r[0], sigma_r[-1])

if shift:
    dt = A.dtype.type
    e = numpy.ones((n, 1), dtype = dt)
    s = numpy.dot(A, e)/n
    A -= numpy.dot(s, e.T)

D = A - numpy.dot(sigma_r*u, vt_r)
err = nla.norm(D, axis = 1)/nla.norm(A, axis = 1)
print('svd error %e' % numpy.amax(err))

B = A.copy()
skl_svd = TruncatedSVD(block_size, tol = tol)
#skl_svd = PCA(block_size, tol = tol)
sigma_skl = numpy.ndarray((0,), dtype = dtype)
vt_skl = numpy.ndarray((0, n), dtype = dtype)
start = time.time()
#skl_svd.fit(A)
while True:
    skl_svd.fit(A)
    s = skl_svd.singular_values_
    vti = skl_svd.components_
    sigma_skl = numpy.concatenate((sigma_skl, s))
    vt_skl = numpy.concatenate((vt_skl, vti))
    stop = time.time()
    time_s = stop - start
    print('%.2f sec: last singular value computed: %e' % (time_s, s[-1]))
    if th == 0:
        break
    if s[-1] <= th:
        break
    print('deflating...')
    A -= numpy.dot(numpy.dot(A, vti.transpose()), vti)
    print('restarting...')
stop = time.time()
time_skl = stop - start
print('sklearn time: %.1e' % time_skl)
#sigma_skl = skl_svd.singular_values_
#vt_skl = skl_svd.components_
#print(sigma_skl)
print(vt_skl.shape)
#print(nla.norm(vt_skl, axis = 1))
n_skl = min(sigma_skl.shape[0], sigma0.shape[0])
if not ptb and n_skl > 0:
    n_skl = min(n_skl, sigma0.shape[0])
    err_vec = vec_err(v0[:,:n_skl], vt_skl.transpose()[:,:n_skl])
    err_val = abs(sigma_skl[:n_skl] - sigma0[:n_skl])
    print('\nmax singular vector error (sklearn): %.1e' % numpy.amax(err_vec))
    print('\nmax singular value error (sklearn): %.1e' % numpy.amax(err_val))
A = B
D = A - numpy.dot(numpy.dot(A, vt_skl.transpose()), vt_skl)
err = nla.norm(D, axis = 1)/nla.norm(A, axis = 1)
print('svd error %e' % numpy.amax(err))

if full:
    print('\n--- solving with scipy.linalg.svd...')
    start = time.time()
    u0, sigma0, vt0 = sla.svd(A, full_matrices = False)#, lapack_driver = 'gesvd')
    stop = time.time()
    time_f = stop - start

if th > 0:
    print('\n--- solving with restarted scipy.sparse.linalg.svds...')
else:
    print('\n--- solving with scipy.sparse.linalg.svds...')

sigma_s = numpy.ndarray((0,), dtype = dtype)
vt_s = numpy.ndarray((0, n), dtype = dtype)

if use_op:
    Ax = lambda x : numpy.dot(A, x)
    ATx = lambda x : numpy.dot(A.T, x)
    opA = LinearOperator(dtype = dtype, shape = (m,n), matmat = Ax, \
                         matvec = Ax, rmatvec = ATx)

start = time.time()

while False:
    if use_op:
        u, s, vti = svds(opA, k = block_size, tol = tol)
    else:
        u, s, vti = svds(A, k = block_size, tol = tol)
    sigma_s = numpy.concatenate((sigma_s, s[::-1]))
    vt_s = numpy.concatenate((vt_s, vti[::-1, :]))
    stop = time.time()
    time_s = stop - start
    print('%.2f sec: last singular value computed: %e' % (time_s, s[0]))
    if th == 0:
        break
    if s[0] <= th:
        break
    print('deflating...')
    A -= numpy.dot(u*s, vti)
    print('restarting...')

stop = time.time()
time_s = stop - start

print('\n%d singular vectors computed' % sigma_s.shape[0])
n_s = min(sigma_s.shape[0], sigma0.shape[0])
if not ptb and n_s > 0:
    err_vec = vec_err(v0[:,:n_s], vt_s.transpose()[:,:n_s])
    err_val = abs(sigma_s[:n_s] - sigma0[:n_s])
    print('\nmax singular vector error (svds): %.1e' % numpy.amax(err_vec))
    print('\nmax singular value error (svds): %.1e' % numpy.amax(err_val))
#else:
#    l = min(n_s, n_r)
#    diff = sigma_r[:l] - sigma_s[:l]
#    print('sigma diff %e' % numpy.amax(diff))
#    #print(sigma[:l])
#    print(sigma_s[0], sigma_s[l - 1])

if full:
    if n_r > 0:
        err_vec = vec_err(vt0.transpose()[:,:n_r], vt_r.transpose()[:,:n_r])
        err_val = abs(sigma_r[:n_r] - sigma0[:n_r])
        print('\nmax singular vector error (raleigh): %.1e' % \
              numpy.amax(err_vec))
        print('\nmax singular value error (raleigh): %.1e' % \
              numpy.amax(err_val))
    if n_s > 0:
        err_vec = vec_err(vt0.transpose()[:,:n_s], vt_s.transpose()[:,:n_s])
        err_val = abs(sigma_s[:n_s] - sigma0[:n_s])
        print('\nmax singular vector error (svds): %.1e' % \
              numpy.amax(err_vec))
        print('\nmax singular value error (svds): %.1e' % \
              numpy.amax(err_val))

print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))
if full:
    print('\n full SVD time: %.1e' % time_f)

print('\ndone')