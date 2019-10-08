# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""
Principal Components update demo.

Performs PCA on a chunk of data, then addds more data and updates Principal 
Components.

Usage: pca_update <data_file> <tolerance> [<q_first>]

data_file : the name of the file containing data matrix X
tolerance : PCA approximation tolerance wanted
q_first   : relative size of the first chunk (default 0.5)
"""

import numpy
import sys

# in case this raleigh package is not pip installed (e.g. cloned from github)
raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.insert(0, raleigh_path)
from raleigh.drivers.pca import pca, pca_error


narg = len(sys.argv)
if narg < 3:
    print('Usage: pca_update <data_file> <tolerance> [<q_first>]')
data = numpy.load(sys.argv[1])
atol = float(sys.argv[2])
q = float(sys.argv[3]) if narg > 3 else 0.5

numpy.random.seed(1) # make results reproducible

m_all = data.shape[0]
if len(data.shape) > 2: # allow for multi-dimensional samples (e.g. images)
    n = numpy.prod(data.shape[1:])
    data = numpy.reshape(data, (m_all, n))
m = min(m_all - 1, max(1, int(q*m_all)))

print('computing PCs for %d data samples...' % m)
mean, trans, comps = pca(data[: m, :], tol=atol, verb=1)
print('%d principal components computed' % trans.shape[1])
em, ef = pca_error(data[: m, :], mean, trans, comps)
print('PCA error: max %.1e, Frobenius %.1e' % (em, ef))

print('\nmore data arrived, updating PCs for %d data samples...' % m_all)
mean, trans, comps = pca(data[m :, :], tol=atol, verb=1, \
    have=(mean, trans, comps))
print('%d principal components computed' % trans.shape[1])
em, ef = pca_error(data, mean, trans, comps)
print('PCA error: max %.1e, Frobenius %.1e' % (em, ef))

print('done')