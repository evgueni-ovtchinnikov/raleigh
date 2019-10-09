# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
""" 
'Smart' Principal Component Analysis test.

Computes PCA approximation of specified accuracy.

PCA allows to approximate m data samples of size n represented by rows of an 
m-by-n matrix X by the rows of the matrix X_k = e a + Y_k Z_k, where 
    e = numpy.ones((m, 1)),
    a = numpy.mean(X, axis=0).reshape((1, n)), 
    the rows of m-by-k matrix Y_k are reduced-features data samples, and 
    the rows of k-by-n matrix Z_k are principal components.

This script computes PCA approximation X_k such that the ratio of the Frobenius
norm of the difference X - X_k to the Frobenius norm of X is not greater than
the user-specified tolerance.

Usage: pca_simple <data_file> <tolerance> <max_pcs> [gpu]

data_file : the name of the file containing data matrix X
tolerance : approximation tolerance wanted
max_pcs   : maximal number of principal components to compute (<1: no limit)
gpu       : run raleigh pca on GPU if this argument is present
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
from raleigh.drivers.pca import pca, pca_error


narg = len(sys.argv)
if narg < 4:
    print('Usage: pca_simple <data_file> <tolerance> <max_pcs> [gpu]')
data = numpy.load(sys.argv[1])
m = data.shape[0]
n = numpy.prod(data.shape[1:])
if len(data.shape) > 2:
    data = numpy.reshape(data, (m, n))
tol = float(sys.argv[2])
mpc = int(sys.argv[3])
arch = 'cpu' if narg < 5 else 'gpu'

numpy.random.seed(1) # make results reproducible

print('\n--- solving with raleigh pca...\n')
start = timeit.default_timer()
mean, trans, comps = pca(data, tol=tol, mpc=mpc, arch=arch, verb=1)
elapsed = timeit.default_timer() - start
ncomp = comps.shape[0]
print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
em, ef = pca_error(data, mean, trans, comps)
print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))

try:
    from sklearn.decomposition import PCA
    # sklearn PCA needs to be given the number k of wanted components;
    # since it is generally unknown in advance, one has to apply deflation:
    # a certain number k of principal components and corresponding 
    # reduced-features samples are computed, and if the PCA error E_k = X - X_k
    # is not small enough, then further k PCs of X (which are PCs of E_k) are 
    # computed etc. until the PCA error falls below the required tolerance
    print('\n--- solving with sklearn PCA...')
    dtype = data.dtype
    # a guess of sufficient number of PCs
    if mpc > 0:
        npc = mpc//2
    else:
        npc = max(1, min(m, n)//10)
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
        if mpc > 0 and pcs >= mpc: # max number of PCs exceeded
            break
        print('deflating...')
        # deflate: subtract computed approximation from data
        data -= numpy.dot(trans, comps)
        print('restarting...')
    elapsed = timeit.default_timer() - start
    ncomp = comps_skl.shape[0]
    print('%d principal components computed in %.2e sec' % (ncomp, elapsed))
    em, ef = pca_error(data0, mean, trans_skl, comps_skl)
    print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))
except:
    pass
print('done')