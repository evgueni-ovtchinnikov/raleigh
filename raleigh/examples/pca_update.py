# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""
Principal Components update demo.

Performs PCA on a chunk of data, then addds more data and updates Principal 
Components.

Usage: pca_update <data_file> <tolerance> [<quota>]

data_file : the name of the file containing data matrix X
tolerance : approximation tolerance wanted
quota     : relative size of the first chunk (default 0.5)
"""

import numpy
import numpy.linalg as nla
import sys

# in case this raleigh package is not pip installed (e.g. cloned from github)
raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.insert(0, raleigh_path)

from raleigh.drivers.pca import pca


def _norm(a, axis):
    return numpy.apply_along_axis(numpy.linalg.norm, axis, a)


def _pca_err(data, mean, trans, comps):
    ones = numpy.ones((data.shape[0], 1), dtype=data.dtype)
    if len(mean.shape) < 2:
        mean = numpy.reshape(mean, (1, comps.shape[1]))
    err = numpy.dot(trans, comps) + numpy.dot(ones, mean) - data
    em = numpy.amax(_norm(err, axis=1))/numpy.amax(_norm(data, axis=1))
    ef = numpy.amax(nla.norm(err, ord='fro'))/numpy.amax(nla.norm(data, ord='fro'))
    return em, ef


narg = len(sys.argv)
if narg < 3:
    print('Usage: pca_update <data_file> <tolerance> [<quota>]')
all_data = numpy.load(sys.argv[1])
atol = float(sys.argv[2])
if narg > 3:
    q = float(sys.argv[3])
else:
    q = 0.5
verb = 1
dtype = all_data.dtype

numpy.random.seed(1) # make results reproducible

shape = all_data.shape
m_all = shape[0]
n = numpy.prod(shape[1:])
all_data = numpy.reshape(all_data, (m_all, n))
m = min(m_all - 1, max(1, int(q*m_all)))
mu = m_all - m
data = all_data[: m, :]
update = all_data[m : m + mu, :]

mean, trans, comps = pca(data, tol=atol, verb=verb)
ncomp = trans.shape[1]
print('%d principal components computed' % ncomp)
em, ef = _pca_err(data, mean, trans, comps)
print('\nPCA error: max %.1e, Frobenius %.1e' % (em, ef))

if mu > 0:
    mean, trans, comps = pca(update, tol=atol, verb=verb, \
                             have=(mean, trans, comps))
    ncomp = trans.shape[1]
    print('%d principal components computed' % ncomp)
    data = numpy.concatenate((data, update))
    em, ef = _pca_err(data, mean, trans, comps)
    print('\nPCA error: max %.1e, Frobenius %.1e' % (em, ef))


print('done')