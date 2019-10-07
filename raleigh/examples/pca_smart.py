# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
""" 'Smart' PCA test.

Computes reduced-feature dataset approximation of specified accuracy.

If sklearn is installed, compares with restarted sklearn.decomposition.PCA.

Usage: pca_simple <data_file> <tolerance> [<n_samples>[, <gpu>]]

data_file : the name of the file containing data
tolerance : approximation tolerance wanted
n_samples : number of data samples to process (optional, all data processed
            by default)
gpu       : run on GPU if this argument is present (value ignored)
"""

import math
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
    ones = numpy.ones((m, 1), dtype=data.dtype)
    if len(mean.shape) < 2:
        mean = numpy.reshape(mean, (1, comps.shape[1]))
    err = numpy.dot(trans, comps) + numpy.dot(ones, mean) - data
    em = numpy.amax(_norm(err, axis=1))/numpy.amax(_norm(data, axis=1))
    ef = numpy.amax(nla.norm(err, ord='fro'))/numpy.amax(nla.norm(data, ord='fro'))
    return em, ef


narg = len(sys.argv)
if narg < 3:
    print('Usage: pca_simple <data_file> <tolerance> [<n_samples>[, <gpu>]]')
file = sys.argv[1]
data = numpy.load(file, mmap_mode='r')
m = data.shape[0]
n = numpy.prod(data.shape[1:])
if len(data.shape) > 2:
    data = numpy.memmap.reshape(data, (m, n))
tol = float(sys.argv[2])
if narg > 3:
    m = min(m, int(sys.argv[3]))
if m < data.shape[0]:
    data = data[:m, :]
arch = 'cpu' if narg < 5 else 'gpu'

numpy.random.seed(1) # make results reproducible

print('\n--- solving with raleigh pca...\n')
start = timeit.default_timer()
mean, trans, comps = pca(data, tol=tol, arch=arch, verb=1)
elapsed = timeit.default_timer() - start
ncomp = comps.shape[0]
print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
em, ef = _pca_err(data, mean, trans, comps)
print('pca error: max %.1e, Frobenius %.1e' % (em, ef))

try:
    from sklearn.decomposition import PCA
    # sklearn PCA needs to be given the number of wanted components;
    # since it is generally unknown in advance, one has to apply deflation:
    # a certain number of components and corresponding reduced-features samples
    # are computed, and if the PCA error 
    #    pca_err = data - mean - reduced_data * components
    # is not small enough, then PCs of data - reduced_data * components are
    # computed recursively
    print('\n--- solving with sklearn PCA...')
    dtype = data.dtype
    npc = max(1, min(m, n)//10) # a guess of sufficient number of PCs
    data0 = data # data will change, save a copy for final error computation
    data = data.copy()
    start = timeit.default_timer()
    skl_pca = PCA(npc)
    sigma_skl = numpy.ndarray((0,), dtype=dtype)
    comps_skl = numpy.ndarray((0, n), dtype=dtype)
    trans_skl = numpy.ndarray((m, 0), dtype=dtype)
    ones = numpy.ones((m, 1), dtype=data.dtype)
    mean = numpy.dot(ones.T, data)/m
    data_s = data - numpy.dot(ones, mean)
    # compute Frobenius norms of data and initial PCA error
    norms = nla.norm(data, axis=1)
    nrms = nla.norm(data_s, axis=1)
    norm_fro2 = numpy.sum(norms*norms)
    err_fro2 = numpy.sum(nrms*nrms)
    while True:
        # compute next portion of PCs
        trans = skl_pca.fit_transform(data)
        sigma = skl_pca.singular_values_
        comps = skl_pca.components_
        sigma_skl = numpy.concatenate((sigma_skl, sigma))
        comps_skl = numpy.concatenate((comps_skl, comps))
        trans_skl = numpy.concatenate((trans_skl, trans), axis=1)
        stop = timeit.default_timer()
        time_s = stop - start
        pcs = comps_skl.shape[0]
        # update Frobenius norm of PCA error
        err_fro2 -= numpy.sum(sigma*sigma)
        err_fro = math.sqrt(err_fro2/norm_fro2)
        print('%.2f sec: last singular value: sigma[%d] = %e, error %.2e' \
            % (time_s, pcs - 1, sigma[-1], err_fro))
        if err_fro < tol: # desired accuracy achieved, quit the loop
            break
        print('deflating...')
        # deflate: subtract computed approximation from data
        data -= numpy.dot(trans, comps)
        print('restarting...')
    elapsed = timeit.default_timer() - start
    ncomp = comps_skl.shape[0]
    print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
    em, ef = _pca_err(data0, mean, trans_skl, comps_skl)
    print('pca error: max %.1e, Frobenius %.1e' % (em, ef))
except:
    pass
print('done')