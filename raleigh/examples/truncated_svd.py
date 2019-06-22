# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)
# This software is distributed under a BSD licence, see ../../LICENSE.txt.

"""Truncated SVD and PCA for a randomly generated matrix.

The singular spectrum of the randomly generated test matrix imitates that
encountered in Principal Component Analyisis or Tensor Network Theory
applications (ascending-ordered singular values quickly approach zero).

Usage:
  truncated_svd [--help | -h | options] <m> <n> <k>

Arguments:
  m  number of rows
  n  number of rows
  k  number of non-zero singular values

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   block CG block size [default: -1]
  -n <rank>, --rank=<rank>   truncated svd rank (negative: unknown a priori, 
                             thsh > 0 will be used as a criterion,
                             if thsh == 0, will run in interactive mode)
                             [default: -1]
  -r <alph>, --alpha=<alph>  singular values decay rate [default: 100]
  -s <thsh>, --thres=<thsh>  singular values threshold [default: 0.01]
  -t <vtol>, --vtol=<vtol>   singular vectors error tolerance [default: 1e-3]
  -v <verb>, --verb=<verb>   verbosity level [default: 0]
  -d, --double   use double precision
  -p, --ptb      add random perturbation to make the matrix full rank
  -P, --pca      compute also PCA
"""

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import numpy
import numpy.linalg as nla
import scipy.linalg as sla
import sys
import time

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.core.solver import Options
from raleigh.drivers.partial_svd import truncated_svd, pca


def norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def random_singular_values(k, sigma, dt):
    s = numpy.random.rand(k).astype(dt)
    s = numpy.sort(s)
    t = numpy.ones(k)*s[0]
    return sigma(s - t)


def random_singular_vectors(m, n, k, dt):
    u = numpy.random.randn(m, k).astype(dt)
    v = numpy.random.randn(n, k).astype(dt)
    u, r = sla.qr(u, mode='economic')
    v, r = sla.qr(v, mode='economic')
    return u, v


def random_matrix_for_svd(m, n, k, sigma, dt):
    s = random_singular_values(min(m, n), sigma, dt)[:k]
    u, v = random_singular_vectors(m, n, k, dt)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a


def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = norm(w, axis = 0)
    return s


if have_docopt:
    args = docopt(__doc__, version=__version__)
    m = int(args['<m>'])
    n = int(args['<n>'])
    k = int(args['<k>'])
    alpha = float(args['--alpha'])
    arch = args['--arch']
    block_size = int(args['--bsize'])
    rank = int(args['--rank'])
    th = float(args['--thres'])
    tol = float(args['--vtol'])
    verb = int(args['--verb'])
    dble = args['--double']
    ptb = args['--ptb']
    do_pca = args['--pca']
else:
    m = 3000
    n = 2000
    k = 1000
    alpha = 100
    arch = 'cpu'
    block_size = -1
    rank = -1
    th = 0.01
    tol = 1e-3
    verb = 0
    dble = False
    ptb = False
    do_pca = True

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
    s = norm(a, axis=0)
    A += a*(sigma0[-1]/s)

# set raleigh solver options
opt = Options()
opt.block_size = block_size
opt.verbosity = verb

print('\n--- solving with truncated_svd...\n')
start = time.time()
u, sigma, vt = truncated_svd(A, opt, nsv=rank, tol=th, vtol=tol, arch=arch)
stop = time.time()
time_tsvd = stop - start
print('\ntruncated svd time: %.1e' % time_tsvd)

nsv = sigma.shape[0]
print('\n%d singular vectors computed' % nsv)
nsv = numpy.sum(sigma > th)
print('%d singular values above threshold' % nsv)
if not ptb and nsv > 0:
    nsv = min(nsv, sigma0.shape[0])
    err_vec = vec_err(v0[:,:nsv], vt.transpose()[:,:nsv])
    err_val = abs(sigma[:nsv] - sigma0[:nsv])
    print('\nmax singular vector error: %.1e' % numpy.amax(err_vec))
    print('max singular value error: %.1e' % numpy.amax(err_val))
D = A - numpy.dot(sigma[:nsv]*u[:, :nsv], vt[:nsv, :])
err = norm(D, axis=1)/norm(A, axis=1)
print('truncation error %.1e' % numpy.amax(err))

if do_pca:
    print('\n--- solving with pca...\n')
    '''
    Principal components do not have to be accurate singular vectors, hence pca
    can compute them faster than truncated_svd in default singular vector accuracy
    (it is this trick that makes scikit-learn pca much faster than much more
    accurate scipy svds).
    '''
    start = time.time()
    mean, trans, comp = pca(A, opt, npc=rank, tol=th, arch=arch)
    stop = time.time()
    time_pca = stop - start
    print('\npca time: %.1e' % time_pca)
    e = numpy.ones((trans.shape[0], 1), dtype=dtype)
    D = A - numpy.dot(trans, comp) - numpy.dot(e, mean)
    err = norm(D, axis=1)/norm(A, axis=1)
    print('\npca error %.1e' % numpy.amax(err))

print('\ndone')
