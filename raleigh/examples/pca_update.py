# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""
Principal Components update demo.

Performs PCA on a chunk of data read from a file, then reads more data and 
updates Principal Components.

Usage:
  pca_update [--help | -h | options] <data>

Arguments:
  data  .npy file containing data as ndarray of two or more dimensions.

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -e <atol>, --aetol=<atol>  PCA approximation error tolerance (<= 0: not set,
                             run in interactive mode) [default: 0]
  -n <nsmp>, --nsamp=<nsmp>  number of samples to use (< 0: all) [default: -1]
  -t <vtol>, --vtol=<vtol>   singular value error tolerance [default: 1e-3]
  -u <usiz>, --usize=<usiz>  update size [default: 1]
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
    atol = float(args['--aetol'])
    mu = int(args['--usize'])
    vtol = float(args['--vtol'])
    verb = int(args['--verb'])
    arch = args['--arch']
    show = args['--show']
else:
    print('\n=== docopt not found, using default options...\n')
    file = sys.argv[1]
    m = -1
    mu = 1
    atol = 0
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

if m < 1 or m > m_all - mu:
    m = m_all - mu

data = all_data[: m, :]
update = all_data[m : m + mu, :].copy()

vmin = numpy.amin(data)
vmax = numpy.amax(data)
print('data range: %e to %e' % (vmin, vmax))

mean, trans, comps = pca(data, tol=atol, arch=arch, verb=verb)
ncon = trans.shape[1]
print('%d principal components computed' % ncon)

# check the accuracy of PCA approximation L R + e a to A
e = numpy.ones((m, 1), dtype=dtype)
err = data - numpy.dot(trans, comps) - numpy.dot(e, mean)
err_max = numpy.amax(_norm(err, axis=1))/numpy.amax(_norm(data, axis=1))
err_f = numpy.amax(nla.norm(err, ord='fro'))/numpy.amax(nla.norm(data, ord='fro'))
print('\npca error: max %.1e, Frobenius %.1e' % (err_max, err_f))

if mu > 0:
    mean, trans, comps = pca(update, tol=atol, arch=arch, verb=verb, \
                             have=(mean, trans, comps))
    ncon = trans.shape[1]
    print('%d principal components computed' % ncon)
    e = numpy.ones((m + mu, 1), dtype=dtype)
    data = numpy.concatenate((data, update))
    err = data - numpy.dot(trans, comps) - numpy.dot(e, mean)
    err_max = numpy.amax(_norm(err, axis=1))/numpy.amax(_norm(data, axis=1))
    err_f = numpy.amax(nla.norm(err, ord='fro'))/numpy.amax(nla.norm(data, ord='fro'))
    print('\npca error: max %.1e, Frobenius %.1e' % (err_max, err_f))

if show and len(shape) == 3:
    # assume data are images
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
        pylab.title('PCA approximation of the image')
        pylab.imshow(pca_image, cmap='gray')
        pylab.show()

print('done')