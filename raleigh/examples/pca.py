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

import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla
import sys
import time

try:
    from sklearn.decomposition import PCA
    have_sklearn = True
except:
    have_sklearn = False

# in case this raleigh package is not pip installed (e.g. cloned from github)
raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.insert(0, raleigh_path)

from raleigh.algebra import verbosity
verbosity.level = 2

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
f_sigma = lambda t: t**(-alpha)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, dtype)
if ptb:
    a = 2*numpy.random.rand(m, n).astype(dtype) - 1
    s = 10*_norm(a, axis=1)
    A += numpy.reshape(sigma0[-1]/s, (m, 1))*a

e = numpy.ones((m, 1), dtype=dtype)
a = numpy.dot(e.T, A)/m

print('\n--- solving with raleigh pca...\n')
start = time.time()
mean_r, trans_r, comps_r = pca(A, npc=rank, tol=atol, norm=norm, svtol=vtol, \
                         arch=arch, verb=verb)
stop = time.time()
time_r = stop - start
ncomp = comps_r.shape[0]
print('\n%d principal components computed in %.1e sec' % (ncomp, time_r))

# check the accuracy of PCA approximation L R + e a to A
D = A - numpy.dot(trans_r, comps_r) - numpy.dot(e, mean_r)
err_max = numpy.amax(_norm(D, axis=1))/numpy.amax(_norm(A, axis=1))
err_f = numpy.amax(nla.norm(D, ord='fro'))/numpy.amax(nla.norm(A, ord='fro'))
print('\npca error: max %.1e, Frobenius %.1e' % (err_max, err_f))

if have_sklearn:

    # sklearn PCA needs to be given the number of wanted components;
    # since it is generally unknown in advance, one has to apply deflation:
    # a certain number of components is computed, and if the PCA error
    # (the difference A - e a - L R) is not small enough, computed L R
    # is subtracted from A and further PCs are computed

    if rank <= 0:
        # a guess of how many PCs are needed
        rank = max(1, min(m, n)//10)
    A0 = A.copy() # A will be changed, save a copy
    norms = nla.norm(A, axis=1)

    if atol > 0:
        print('\n--- solving with restarted sklearn.decomposition.PCA...\n')
    else:
        print('\n--- solving with sklearn.decomposition.PCA...\n')
    start = time.time()
    skl_pca = PCA(rank)
    sigma_skl = numpy.ndarray((0,), dtype=dtype)
    comps_skl = numpy.ndarray((0, n), dtype=dtype)
    trans_skl = numpy.ndarray((m, 0), dtype=dtype)
    As = A - numpy.dot(e, a)
    if norm == 'f':
        # compute Frobenius norms of A and initial PCA error
        nrms = nla.norm(As, axis=1)
        norm_fro2 = numpy.sum(norms*norms)
        err_fro2 = numpy.sum(nrms*nrms)
    while True:
        # compute next portion of PCs
        skl_pca.fit(A)
        sigma = skl_pca.singular_values_
        comps = skl_pca.components_
        trans = numpy.dot(As, comps.T)
        if sigma_skl.shape[0] == 0:
            sigma_max = sigma[0]
        sigma_skl = numpy.concatenate((sigma_skl, sigma))
        comps_skl = numpy.concatenate((comps_skl, comps))
        trans_skl = numpy.concatenate((trans_skl, trans), axis=1)
        stop = time.time()
        time_s = stop - start
        pcs = comps_skl.shape[0]
        err_sgm = sigma[-1]/sigma_max
        if norm == 'f':
            # update Frobenius norm of PCA error
            err_fro2 -= numpy.sum(sigma*sigma)
            err_fro = math.sqrt(err_fro2/norm_fro2)
        print('%.2f sec: last singular value: sigma[%d] = %e = %.2e*sigma[0]' \
            % (time_s, pcs - 1, sigma[-1], err_sgm))
        if atol <= 0 or norm not in ['f', 'm'] and err_sgm < atol \
            or norm == 'f' and err_fro < atol:
            # desired accuracy achieved, quit the loop
            break
        print('deflating...')
        # subtract computed approximation from A
        A -= numpy.dot(trans, comps)
        # compute PCA errors
        As = A - numpy.dot(e, a)
        errs = nla.norm(As, axis = 1)
        err_max = numpy.amax(errs)/numpy.amax(norms)
        err_ave = math.sqrt(numpy.sum(errs*errs)/numpy.sum(norms*norms))
        print('PCA error max l2: %.2e, Frobenius: %.2e, max sv: %.2e' % \
            (err_max, err_ave, err_sgm))
        if norm == 'f':
            err = err_ave
        elif norm == 'm':
            err = err_max
        else:
            err = err_sgm
        if err <= atol:
            # desired accuracy achieved, quit the loop
            break
        print('restarting...')
    stop = time.time()
    time_k = stop - start
    print('\n---\nsklearn time: %.1e' % time_k)

    # check the accuracy of PCA approximation
    As = A0 - numpy.dot(e, a)
    As -= numpy.dot(trans_skl, comps_skl)
    errs = nla.norm(As, axis = 1)
    err_max = numpy.amax(errs)/numpy.amax(norms)
    err_ave = nla.norm(As, ord='fro')/nla.norm(A0, ord='fro')
    print('PCA error max l2: %.2e, Frobenius: %.2e' % (err_max, err_ave))

print('\ndone')