# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''Principal Component Analysis of 2D images.

Computes Principal Components for a set of 2D images incrementally until
the error of PCA approximation falls below the required tolerance.

Note on Linear Algebra style terminology adopted here:
Due to a specific nature of this PCA application, principal components are
referred to as 'eigenimages'. Each image from the original data set is
approximated by a linear combination of eigenimages, hence we refer to the
coefficients of these linear combinations as 'coordinates' (of images in
the orthonormal basis of eigenimages) - in some other PCA software these
are referred to as 'reduced features data'.

If you have docopt:

Usage:
  compute_eigenimages [--help | -h | options] <images> <err_tol>

Arguments:
  images   .npy file containing images as ndarray of dimensions (ni, ny, nx)
  err_tol  image approximation error tolerance

Options:
  -n <nim> , --nimgs=<nim>   number of images to use (< 0: all) [default: -1]
  -a <arch>, --arch=<arch>   architecture [default: cpu]

If you do not have docopt:

  compute_eigenimages <images> <err_tol> [gpu]
'''

try:
    from docopt import docopt
    have_docopt = True
except:
    have_docopt = False

import math
import numpy
import pylab
import sys
import timeit

from raleigh.interfaces.pca import pca
from raleigh.core.solver import Options
from raleigh.interfaces.truncated_svd import UserStoppingCriteria


def _norm(a, axis):
    return numpy.apply_along_axis(numpy.linalg.norm, axis, a)


if have_docopt:
    __version__ = '0.1.0'
    args = docopt(__doc__, version=__version__)
    file = args['<images>']
    err_tol = float(args['<err_tol>'])
    ni = int(args['--nimgs'])
    arch = args['--arch']
else:
    narg = len(sys.argv)
    if narg < 3:
        print('Usage: compute_eigenimages <images> <err_tol> [gpu]')
        exit()
    file = sys.argv[1]
    err_tol = float(sys.argv[2])
    arch = 'cpu' if narg < 4 else 'gpu!'
    ni = -1

numpy.random.seed(1) # make results reproducible

all_images = numpy.load(file)

m_all, ny, nx = all_images.shape
n = nx*ny

if ni < 0 or ni > m_all:
    ni = m_all

if ni < m_all:
    print('using first %d images only...' % ni)
    m = ni
    images = all_images[:m,:,:]
else:
    print('using all %d images...' % m_all)
    m = m_all
    images = all_images

images = numpy.reshape(images, (m, n))

start = timeit.default_timer()
mean, coord, eigim = pca(images, tol=err_tol, arch=arch, verb=1)
elapsed_time = timeit.default_timer() - start
ncon = eigim.shape[0]
print('%d eigenimages computed in %.2e sec' % (ncon, elapsed_time))

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= m:
        break
    pylab.figure()
    pylab.title('image %d' % i)
    img = images[i,:]
    image = numpy.reshape(img, (ny, nx))
    pylab.imshow(image, cmap='gray')
    img = numpy.dot(coord[i,:], eigim) + mean
    pca_image = numpy.reshape(img, (ny, nx))
    pylab.figure()
    pylab.title('PCA approximation of the image')
    pylab.imshow(pca_image, cmap='gray')
    pylab.show()

print('saving...')
mean = numpy.reshape(mean, (ny, nx))
eigim = numpy.reshape(eigim, (ncon, ny, nx))
numpy.savez('eigenimages', eigim=eigim, coord=coord, mean=mean)

print('done')
