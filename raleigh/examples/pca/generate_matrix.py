# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
'''Generates a data matrix from randomly generated singular vectors and 
   singular values of user-controlled behaviour.

   The generated matrix is saved to data.npy, and singular values and vectors
   to svd.npz.

    The singular spectrum of the randomly generated test matrix imitates that
    encountered in Principal Component Analyisis of images from lfw dataset
    (see http://vis-www.cs.umass.edu/lfw).

Usage:
  generate_matrix [--help | -h | options] <m> <n> <r>

Arguments:
  m  number of rows in A
  n  number of columns in A
  r  rank of A (use r << min(m, n) for fast generation)

Options:
  -a <alpha>, --alpha=<alpha>  singular values decay rate [default: 0.75]
  -d, --double   use double precision
  -p, --ptb      add random perturbation to make the matrix full rank
  -s, --pca      generate matrix with shift-to-zero-average invariant singular
                 values (save for the first), to be used for testing PCA
'''

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import numpy
import numpy.linalg as nla
import scipy.linalg as sla


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def random_singular_values(k, sigma, dt):
    s = numpy.random.rand(k).astype(dt)
    s = numpy.sort(s)
    return sigma(s)


def random_singular_vectors(m, n, k, dt, pca):
    u = numpy.random.randn(m, k).astype(dt)
    if pca:
        u[:, 0] = 1.0
    v = numpy.random.randn(n, k).astype(dt)
    u, r = sla.qr(u, mode='economic')
    v, r = sla.qr(v, mode='economic')
    return u, v


def random_matrix_for_svd(m, n, k, sigma, dt, pca=False):
    s = random_singular_values(min(m, n), sigma, dt)[:k]
    u, v = random_singular_vectors(m, n, k, dt, pca)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a


if have_docopt:
    args = docopt(__doc__, version=__version__)
    m = int(args['<m>'])
    n = int(args['<n>'])
    k = int(args['<r>'])
    alpha = float(args['--alpha'])
    dble = args['--double']
    ptb = args['--ptb']
    pca = args['--pca']
else:
    narg = len(sys.argv)
    if narg < 4:
        usage = 'Usage: generate_matrix <samples> <features> <rank>'
        raise SystemExit(usage)
    m = sys.argv[1]
    n = sys.argv[2]
    k = sys.argv[3]
    alpha = 0.75
    dble = False
    ptb = False
    pca = True

numpy.random.seed(1) # make results reproducible

if dble:
    dtype = numpy.float64
else:
    dtype = numpy.float32

print('\n--- generating the matrix...')
f_sigma = lambda t: t**(-alpha)
sigma, u, v, A = random_matrix_for_svd(m, n, k, f_sigma, dtype, pca)
if ptb:
    a = 2*numpy.random.rand(m, n).astype(dtype) - 1
    s = 10*_norm(a, axis=1)
    A += numpy.reshape(sigma[-1]/s, (m, 1))*a

e = numpy.ones((m, 1), dtype=dtype)

print('\n--- saving...')
numpy.save('data.npy', A)
numpy.savez('svd', sigma=sigma, left=u, right=v)

print('done')
