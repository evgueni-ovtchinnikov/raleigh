# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""PCA for a randomly generated matrix.

Randomly generates m by n data matrix A (m data samples n features each),
and computes m by k matrix L and k by n matrix R such that k <= min(m, n),
and the product L R approximates A - e a, where e = numpy.ones((m, 1))
and a = numpy.mean(A, axis=0).

The rows of R (principal components) are orhonormal, the columns of L
(reduced features) are in the descending order of their norms.

The singular spectrum of the randomly generated test matrix imitates that
encountered in Principal Component Analyisis of images from lfw dataset
(see http://vis-www.cs.umass.edu/lfw).

Usage:
  pca [--help | -h | options] <m> <n> <r>

Arguments:
  m  number of rows in A
  n  number of columns in A
  r  rank of A (use r << min(m, n) for fast generation)

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   block CG block size (<0: auto) [default: -1]
  -e <atol>, --atol=<atol >  pca approximation error [default: 0]
  -m <nrm> , --norm=<nrm>    approximation error norm to use:
                             f: Frobenius norm of the error matrix
                             m: max l2 norm of the error matrix row
                             s: max singular value of the error matrix
                             (each norm relative to the respective norm of
                             the data matrix)
                             [default: f]
  -n <rank>, --rank=<rank>   truncated svd rank (negative: unknown a priori, 
                             atol > 0 will be used as a criterion,
                             if atol == 0, will run in interactive mode)
                             [default: -1]
  -r <alph>, --alpha=<alph>  singular values decay rate [default: 1]
  -t <vtol>, --vtol=<vtol>   singular values error tolerance relative to the
                             largest singular value [default: 1e-3]
  -v <verb>, --verb=<verb>   verbosity level [default: 0]
  -d, --double   use double precision
  -p, --ptb      add random perturbation to make the matrix full rank
"""

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

#import math
import numpy
import numpy.linalg as nla
import pylab
import scipy.linalg as sla
import sys
import time

# in case this raleigh package is not pip installed (e.g. cloned from github)
raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.insert(0, raleigh_path)

from raleigh.algebra import verbosity
verbosity.level = 2

from raleigh.core.solver import Options
from raleigh.drivers.pca import pca


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def random_singular_values(k, sigma, dt):
    s = numpy.random.rand(k).astype(dt)
    s = numpy.sort(s)
    return sigma(s)


def random_singular_vectors(m, n, k, dt):
    u = numpy.random.randn(m, k).astype(dt)
    u[:, 0] = 1.0
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
    s = _norm(w, axis = 0)
    return s


if have_docopt:
    args = docopt(__doc__, version=__version__)
    m = int(args['<m>'])
    n = int(args['<n>'])
    k = int(args['<r>'])
    alpha = float(args['--alpha'])
    arch = args['--arch']
    block_size = int(args['--bsize'])
    norm = args['--norm']
    rank = int(args['--rank'])
    atol = float(args['--atol'])
    vtol = float(args['--vtol'])
    verb = int(args['--verb'])
    dble = args['--double']
    ptb = args['--ptb']
else:
    print('\n=== docopt not found, using default options...\n')
    m = 3000
    n = 2000
    k = 1000
    alpha = 100
    arch = 'cpu'
    block_size = -1
    rank = -1
    norm = 'f'
    atol = 0
    vtol = 1e-3
    verb = 0
    dble = False
    ptb = False

numpy.random.seed(1) # make results reproducible

if dble:
    dtype = numpy.float64
else:
    dtype = numpy.float32

print('\n--- generating the matrix...')
f_sigma = lambda t: t**(-alpha) #.astype(dtype)
#f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, dtype)
if ptb:
    a = 2*numpy.random.rand(m, n).astype(dtype) - 1
    s = 10*_norm(a, axis=1)
    A += numpy.reshape(sigma0[-1]/s, (m, 1))*a

pylab.figure()
pylab.plot(numpy.arange(1, k + 1, 1), sigma0)
pylab.xscale('log')
pylab.yscale('log')
pylab.grid()
pylab.title('singular values')
pylab.show()

e = numpy.ones((m, 1), dtype=dtype)
a = numpy.dot(e.T, A)/m

print('\n--- solving with raleigh pca...\n')
# set raleigh solver options
opt = Options()
opt.block_size = block_size
#opt.verbosity = verb
# do pca
start = time.time()
mean, trans, comps = pca(A, opt, npc=rank, tol=atol, norm=norm, svtol=vtol, \
                         arch=arch, verb=verb)
stop = time.time()
time_pca = stop - start
print('\npca time: %.1e' % time_pca)
D = A - numpy.dot(trans, comps) - numpy.dot(e, mean)
err_max = numpy.amax(_norm(D, axis=1))/numpy.amax(_norm(A, axis=1))
err_f = numpy.amax(nla.norm(D, ord='fro'))/numpy.amax(nla.norm(A, ord='fro'))
print('\npca error: max %.1e, Frobenius %.1e' % (err_max, err_f))

print('\ndone')