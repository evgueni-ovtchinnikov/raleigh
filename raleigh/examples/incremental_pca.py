# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""
Incremental Principal Component Analysis demo.

Principal components are computed incrementally, processing a number 
(batch size) of data samples at a time.

For comparison, the same amount of principal components is computed also by 
scikit-learn IncrementalPCA.

Usage: incremental_pca <data_file> <batch_size> <tolerance> <max_pcs> [gpu]

Arguments:

data_file  : the name of the file containing data matrix
batch_size : batch size
tolerance  : approximation tolerance wanted
max_pcs    : maximal number of principal components to compute (<1: no limit)
gpu        : run raleigh pca on GPU if this argument is present
"""

import numpy
import sys
import timeit

try:
    from sklearn.decomposition0 import IncrementalPCA
    have_sklearn = True
except:
    have_sklearn = False

# in case this raleigh package is not pip installed (e.g. cloned from github)
raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.insert(0, raleigh_path)
from raleigh.drivers.pca import pca, pca_error


narg = len(sys.argv)
if narg < 5:
    print('Usage: incremental_pca <data_file> <batch_size> <tolerance> <max_pcs> [gpu]')
data = numpy.load(sys.argv[1], mmap_mode='r')
atol = float(sys.argv[3])
batch_size = int(sys.argv[2])
mpc = int(sys.argv[4])
arch = 'cpu' if narg < 6 else 'gpu'

numpy.random.seed(1) # make results reproducible

dtype = data.dtype
m = data.shape[0]
if len(data.shape) > 2:
    n = numpy.prod(data.shape[1:])
    data = numpy.memmap.reshape(data, (m, n))

print('\n--- solving with raleigh pca...\n')
start = timeit.default_timer()
mean, trans, comps = pca(data, mpc=mpc, tol=atol, arch=arch, verb=1, \
    batch_size=batch_size)
elapsed = timeit.default_timer() - start
ncomp = comps.shape[0]
print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
em, ef = pca_error(data, mean, trans, comps)
print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))

if have_sklearn and arch == 'cpu':
    npc = ncomp
    if npc > batch_size:
        batch_size = npc
    print('\n--- solving with sklearn.decomposition.IncrementalPCA...\n')
    start = timeit.default_timer()
    skl_ipca = IncrementalPCA(n_components=npc, batch_size=batch_size)
    skl_trans = skl_ipca.fit_transform(data)
    skl_comps = skl_ipca.components_
    skl_mean = skl_ipca.mean_
    stop = timeit.default_timer()
    elapsed = stop - start
    ncomp = skl_comps.shape[0]
    print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
    em, ef = pca_error(data, skl_mean, skl_trans, skl_comps)
    print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))

print('\ndone')