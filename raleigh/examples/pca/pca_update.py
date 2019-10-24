# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""
Principal Components update demo.

Performs PCA on a chunk of data, then addds more data and updates Principal 
Components.

Usage: pca_update <data_file> <tolerance> <q_first> <max_pcs> [gpu]

data_file : the name of the file containing data matrix X
tolerance : PCA approximation tolerance wanted
q_first   : relative size of the first chunk (default 0.5)
max_pcs   : maximal number of principal components to compute (<1: no limit)
"""

import numpy
import sys
import timeit

from raleigh.drivers.pca import pca, pca_error


narg = len(sys.argv)
if narg < 5:
    usage = \
          'Usage: pca_update <data_file> <tolerance> <q_first> <max_pcs> [gpu]'
    raise SystemExit(usage)
data = numpy.load(sys.argv[1])
atol = float(sys.argv[2])
q = float(sys.argv[3])
mpc = int(sys.argv[4])
arch = 'cpu' if narg < 6 else 'gpu'

numpy.random.seed(1) # make results reproducible

m_all = data.shape[0]
n = data.shape[1]
if len(data.shape) > 2: # allow for multi-dimensional samples (e.g. images)
    n = numpy.prod(data.shape[1:])
    data = numpy.reshape(data, (m_all, n))
m = min(m_all, max(1, int(q*m_all)))

print('computing PCs for %d data samples...' % m)
start = timeit.default_timer()
mean, trans, comps = pca(data[: m, :], tol=atol, mpc=mpc, verb=1, arch=arch)
elapsed = timeit.default_timer() - start
ncomp = comps.shape[0]
print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
em, ef = pca_error(data[: m, :], mean, trans, comps)
print('PCA error: max %.1e, Frobenius %.1e' % (em, ef))

if m < m_all:
    print('\nmore data arrived, updating PCs for %d data samples...' % m_all)
    start = timeit.default_timer()
    if atol == 0 and mpc < 1: # interactive mode not helpful in update
        mpc = comps.shape[0]
    mean, trans, comps = pca(data[m :, :], mpc=mpc, tol=atol, verb=0, arch=arch, \
        have=(mean, trans, comps))
    elapsed = timeit.default_timer() - start
    ncomp = comps.shape[0]
    print('%d principal components updated in %.2e sec' % (ncomp, elapsed))
    em, ef = pca_error(data, mean, trans, comps)
    print('PCA error: max %.1e, Frobenius %.1e' % (em, ef))

print('done')
