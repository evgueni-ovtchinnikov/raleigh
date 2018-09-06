# -*- coding: utf-8 -*-
"""Comparisons with svds.

Usage:
  svds [--help | -h | options] <m> <n> <k>

Arguments:
  m  number of rows
  n  number of rows
  k  number of non-zero singular values

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   block CG block size [default: 64]
  -r <alph>, --alpha=<alph>  singular values decay rate [default: 0.01]
  -s <ths> , --thresh=<ths>  singular values threshold [default: 0.01]
  -t <tol> , --svtol=<tol>   singular vector error tolerance [default: 1e-2]
  -f, --full  compute full SVD too (using scipy.linalg.svd)
  -p, --ptb   add random perturbation to make the matrix full rank

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)
#print(args)

m = int(args['<m>'])
n = int(args['<n>'])
k = int(args['<k>'])
alpha = float(args['--alpha'])
arch = args['--arch']
block_size = int(args['--bsize'])
#err_tol = float(args['--svderr'])
th = float(args['--thresh'])
svec_tol = float(args['--svtol'])
full = args['--full']
ptb = args['--ptb']

import numpy
#import numpy.linalg as nla
import scipy.linalg as sla
import sys
import time

from scipy.sparse.linalg import svds

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from random_matrix_for_svd import random_matrix_for_svd
from raleigh.solver import Options
from raleigh.ndarray.svd import partial_svd #, PSVDErrorCalculator

class MyStoppingCriteria:
    def __init__(self, a):
#        self.err_tol = 0.0
        self.th = 1.0
        self.m = -1
        self.iteration = -1
#        self.err_calc = PSVDErrorCalculator(a)
#        self.norms = self.err_calc.norms
#        self.err = self.norms
#        print('max data norm: %e' % numpy.amax(self.err))
#    def set_error_tolerance(self, err_tol):
#        self.err_tol = err_tol
    def set_threshold(self, th):
        self.th = th*th
    def set_how_many(self, m):
        self.m = m
    def satisfied(self, solver):
        self.iteration = solver.iteration
        if solver.rcon < 1:
            return False
        vmin = numpy.amin(solver.eigenvalues)
        done = vmin < self.th or self.m > 0 and solver.rcon >= self.m
#        self.err = self.err_calc.update_errors()
#        errs = (numpy.amax(self.err), numpy.amax(self.err/self.norms))
#        print('max err: abs %e, rel %e' % errs)
#        done = errs[1] <= self.err_tol or self.m > 0 and solver.rcon >= self.m
        return done

def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = numpy.linalg.norm(w, axis = 0)
    return s

numpy.random.seed(1) # make results reproducible

dtype = numpy.float32

print('\n--- generating the matrix...')
#alpha = 0.01
f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, dtype)
if ptb:
    a = 2*numpy.random.rand(m, n).astype(dtype) - 1
    s = numpy.linalg.norm(a, axis = 0)
    A += a*(sigma0[-1]/s)

if block_size < 1:
    b = max(1, min(m, n)//100)
#    block_size = ((b - 1)//8 + 1)*8
    block_size = 32
    while block_size < b:
        block_size += 32
    print('using block size %d' % block_size)

print('\n--- solving with raleigh.ndarray.partial_svd...')

# set raleigh solver options
opt = Options()
if th > 0:
    opt.block_size = block_size
opt.max_iter = 400
opt.verbosity = -1
opt.convergence_criteria.set_error_tolerance \
    ('kinematic eigenvector error', svec_tol)
opt.stopping_criteria = MyStoppingCriteria(A)
if th > 0:
    opt.stopping_criteria.set_threshold(th)
else:
    opt.stopping_criteria.set_threshold(0)
#    opt.stopping_criteria.set_how_many(block_size)
#opt.stopping_criteria.set_error_tolerance(err_tol)

start = time.time()
if th > 0:
    sigma, u, vt = partial_svd(A, opt, arch = arch)
else:
    sigma, u, vt = partial_svd(A, opt, nsv = block_size, arch = arch)
stop = time.time()
time_r = stop - start
iter_r = opt.stopping_criteria.iteration
print('raleigh time: %.1e' % time_r)

print('\n%d singular vectors computed' % sigma.shape[0])
n_r = min(sigma.shape[0], sigma0.shape[0])
if not ptb:
    err_vec = vec_err(v0[:,:n_r], vt.transpose()[:,:n_r])
    err_val = abs(sigma[:n_r] - sigma0[:n_r])
    #B = A - numpy.dot(u*sigma, vt)
    #err = nla.norm(B, axis = 1)/nla.norm(A, axis = 1)
    #print('\nmax SVD error: %e' % numpy.amax(err))
    print('\nmax singular vector error (raleigh): %.1e' % numpy.amax(err_vec))
    print('\nmax singular value error (raleigh): %.1e' % numpy.amax(err_val))

if full:
    print('\n--- solving with scipy.linalg.svd...')
    start = time.time()
    u, sigma, vt = sla.svd(A, full_matrices = False)#, lapack_driver = 'gesvd')
    stop = time.time()
    time_f = stop - start
#    print(sigma[-100:-1])
#    print(sigma[k : k + 100])
    print('\n full SVD time: %.1e' % time_f)

if th > 0:
    print('\n--- solving with restarted scipy.sparse.linalg.svds...')
else:
    print('\n--- solving with scipy.sparse.linalg.svds...')

sigma = numpy.ndarray((0,), dtype = dtype)
vt = numpy.ndarray((0, n), dtype = dtype)
#normA = numpy.amax(nla.norm(A, axis = 1))

start = time.time()

while True:
    u, s, vti = svds(A, k = block_size, tol = svec_tol)
    #print(s[::-1])
    #print(s[-1])
    sigma = numpy.concatenate((sigma, s[::-1]))
    vt = numpy.concatenate((vt, vti[::-1, :]))
    print('last singular value computed: %e' % s[0])
    if th == 0:
        break
    if s[0] <= th:
        break
    print('deflating...')
    A -= numpy.dot(u*s, vti)
#    errs = numpy.amax(nla.norm(A, axis = 1))/normA
#    print('max SVD error: %.1e' % errs)
#    if errs <= err_tol:
#        break
    print('restarting...')

stop = time.time()
time_s = stop - start
#print(sigma)

print('\n%d singular vectors computed' % sigma.shape[0])
n_s = min(sigma.shape[0], sigma0.shape[0])
if not ptb:
    err_vec = vec_err(v0[:,:n_s], vt.transpose()[:,:n_s])
    err_val = abs(sigma[:n_s] - sigma0[:n_s])
    print('\nmax singular vector error (svds): %.1e' % numpy.amax(err_vec))
    print('\nmax singular value error (svds): %.1e' % numpy.amax(err_val))

print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))
if full:
    print('\n full SVD time: %.1e' % time_f)

print('\ndone')
