# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
'''Simple PCA test.

Computes a given number of principal components of a dataset.

If sklearn is installed, compares with sklearn.decomposition.PCA.

Usage: pca_simple <data_file> <n_components> [gpu]

data_file    : the name of the file containing data
n_components : number of principal components wanted
gpu          : run raleigh pca on GPU if this argument is present
'''

import numpy
import numpy.linalg as nla
import sys
import timeit

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
    print('Usage: pca_simple <data_file> <n_components> [gpu]')
data = numpy.load(sys.argv[1])
npc = int(sys.argv[2])
m = data.shape[0]
if len(data.shape) > 2:
    n = numpy.prod(data.shape[1:])
    data = numpy.reshape(data, (m, n))
arch = 'cpu' if narg < 4 else 'gpu'

numpy.random.seed(1) # make results reproducible

print('\n--- solving with raleigh pca...\n')
start = timeit.default_timer()
mean, trans, comps = pca(data, npc=npc, arch=arch)
elapsed = timeit.default_timer() - start
ncomp = comps.shape[0]
print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
em, ef = _pca_err(data, mean, trans, comps)
print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))

try:
    from sklearn.decomposition import PCA
    print('\n--- solving with sklearn PCA...')
    start = timeit.default_timer()
    skl_pca = PCA(npc)
    trans = skl_pca.fit_transform(data)
    elapsed = timeit.default_timer() - start
    comps = skl_pca.components_
    ncomp = comps.shape[0]
    print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
    em, ef = _pca_err(data, mean, trans, comps)
    print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))
except:
    pass
print('done')