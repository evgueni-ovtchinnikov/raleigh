# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""
Incremental Principal Component Analysis demo.

PCs are computed incrementally, processing a number (batch size) of data 
samples at a time.

Usage:
  incremental_pca [--help | -h | options] <data>

Arguments:
  data  .npy file containing data as ndarray of two or more dimensions.

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <bsze>, --bsize=<bsze>  batch size (<0: all data) [default: -1]
  -c <comp>, --ncomp=<comp>  number of components to compute (< 0: not known,
                             compute based on the PCA approximation error 
                             tolerance or interactively) [default: -1]
  -e <atol>, --aetol=<atol>  PCA approximation error tolerance (<= 0: not set,
                             run in interactive mode) [default: 0]
  -n <nsmp>, --nsamp=<nsmp>  number of samples to use (< 0: all) [default: -1]
  -t <vtol>, --vtol=<vtol>   singular value error tolerance [default: 1e-3]
  -v <verb>, --verb=<verb>   verbosity level [default: 0]
  -s, --show  display data as images (data must be 3D)

Created on Wed Sep 18 13:45:14 2019
"""

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import numpy
import numpy.linalg as nla
import sys
import time

try:
    from sklearn.decomposition import IncrementalPCA
    have_sklearn = True
except:
    have_sklearn = False

# in case this raleigh package is not pip installed (e.g. cloned from github)
raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.insert(0, raleigh_path)

from raleigh.drivers.pca import pca


def _norm(a, axis):
    return numpy.apply_along_axis(numpy.linalg.norm, axis, a)


if have_docopt:
    __version__ = '0.1.0'
    args = docopt(__doc__, version=__version__)
    file = args['<data>']
    m = int(args['--nsamp'])
    npc = int(args['--ncomp'])
    atol = float(args['--aetol'])
    batch_size = int(args['--bsize'])
    vtol = float(args['--vtol'])
    verb = int(args['--verb'])
    arch = args['--arch']
    show = args['--show']
else:
    print('\n=== docopt not found, using default options...\n')
    file = sys.argv[1]
    m = -1
    npc = -1
    atol = 0
    batch_size = -1
    vtol = 1e-3
    verb = 0
    arch = 'cpu'
    show = False

dtype = numpy.float32

numpy.random.seed(1) # make results reproducible

all_data = numpy.load(file, mmap_mode='r')

shape = all_data.shape
m_all = shape[0]
n = numpy.prod(shape[1:])
all_data = numpy.memmap.reshape(all_data, (m_all, n))

if m < 1 or m > m_all:
    m = m_all

if m < m_all:
    data = all_data[:m, :]
else:
    data = all_data

if batch_size < 0:
    batch_size = m

e = numpy.ones((m, 1), dtype=dtype)

vmin = numpy.amin(data)
vmax = numpy.amax(data)
print('data range: %e to %e' % (vmin, vmax))

start = time.time()
mean, trans, comps = pca(data, npc=npc, tol=atol, arch=arch, verb=verb, \
    batch_size=batch_size)
stop = time.time()
elapsed_time = stop - start

sigma = _norm(trans, axis=0)
ncon = sigma.shape[0]
print('%d principal components computed' % ncon)
if atol > 0 or npc > 0:
    print('elapsed time: %.2e' % elapsed_time)

# check the accuracy of PCA approximation L R + e a to A
w = numpy.dot(trans, comps) + numpy.dot(e, mean)
w -= data
err_max = numpy.amax(_norm(w, axis=1))/numpy.amax(_norm(data, axis=1))
err_f = numpy.amax(nla.norm(w, ord='fro')) \
        /numpy.amax(nla.norm(data, ord='fro'))
print('pca error: max %.1e, Frobenius %.1e' % (err_max, err_f))
del w

if have_sklearn:
    if npc < 0:
        npc = ncon
    if npc > batch_size:
        batch_size = npc
    print('\n--- solving with sklearn.decomposition.IncrementalPCA...')
    start = time.time()
    skl_ipca = IncrementalPCA(n_components=npc, batch_size=batch_size)
    skl_trans = skl_ipca.fit_transform(data)
    skl_comps = skl_ipca.components_
    skl_mean = numpy.reshape(skl_ipca.mean_, (1, n))
    stop = time.time()
    time_skl = stop - start
    print('elapsed time: %.1e' % time_skl)
    pcs = skl_comps.shape[0]
    skl_err = data - numpy.dot(skl_trans, skl_comps) - numpy.dot(e, skl_mean)
    err_max = numpy.amax(_norm(skl_err, axis=1)) \
                /numpy.amax(_norm(data, axis=1))
    err_f = numpy.amax(nla.norm(skl_err, ord='fro')) \
            /numpy.amax(nla.norm(data, ord='fro'))
    print('pca error: max %.1e, Frobenius %.1e' % (err_max, err_f))

if show and len(shape) == 3:
    # data samples are 2D: assume they are images
    import pylab
    ny = shape[1]
    nx = shape[2]
    while True:
        i = int(input('image number (negative to exit): '))
        if i < 0 or i >= m:
            break
        pylab.figure()
        pylab.title('image %d' % i)
        img = data[i,:]
        image = numpy.reshape(img, (ny, nx))
        pylab.imshow(image, cmap='gray')
        img = numpy.dot(trans[i,:], comps) + mean
        pca_image = numpy.reshape(img, (ny, nx))
        pylab.figure()
        pylab.title('PCA approximation of the image (raleigh)')
        pylab.imshow(pca_image, cmap='gray')
        img = numpy.dot(skl_trans[i,:], skl_comps) + skl_mean
        pca_image = numpy.reshape(img, (ny, nx))
        pylab.figure()
        pylab.title('PCA approximation of the image (sklearn)')
        pylab.imshow(pca_image, cmap='gray')
        pylab.show()

print('done')