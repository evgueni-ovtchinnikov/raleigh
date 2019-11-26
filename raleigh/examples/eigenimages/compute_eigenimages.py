# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''Principal Component Analysis of 2D images.

Computes Principal Components for a set of 2D images in small portions until
stopped by user's entering 'n' in answer to 'more?' or the error of PCA
approximation falls below the tolerance.

Note on Linear Algebra style terminology adopted here:
Due to a specific nature of this PCA application, principal components are
referred to as 'eigenimages'. Each image from the original data set is
approximated by a linear combination of eigenimages, hence we refer to the
coefficients of these linear combinations as 'coordinates' (of images in
the orthonormal basis of eigenimages) - in some other PCA software these
are referred to as 'reduced features data'.

Usage:
  compute_eigenimages [--help | -h | options] <images>

Arguments:
  images  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -n <nim> , --nimgs=<nim>   number of images to use (< 0: all) [default: -1]
  -e <err> , --imerr=<err>   image approximation error tolerance (<= 0: not set,
                             run in interactive mode) [default: 0]
  -a <arch>, --arch=<arch>   architecture [default: cpu]
'''

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import numpy
import pylab
import sys
import timeit

from raleigh.interfaces.pca import pca

try:
    input = raw_input
except NameError:
    pass


def _norm(a, axis):
    return numpy.apply_along_axis(numpy.linalg.norm, axis, a)


if have_docopt:
    __version__ = '0.1.0'
    args = docopt(__doc__, version=__version__)
    file = args['<images>']
    ni = int(args['--nimgs'])
    err_tol = float(args['--imerr'])
    arch = args['--arch']
else:
    narg = len(sys.argv)
    if narg < 3:
        print('Usage: compute_eigenimages <images> <pca_err> [gpu]')
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

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))

start = timeit.default_timer()
mean, coord, eigim = pca(images, tol=err_tol, arch=arch, verb=1)
elapsed_time = timeit.default_timer() - start

sigma = _norm(coord, axis=0)
ncon = sigma.shape[0]
print('%d eigenimages computed' % ncon)
if err_tol > 0:
    print('elapsed time: %.2e' % elapsed_time)

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
