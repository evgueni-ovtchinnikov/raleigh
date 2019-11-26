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

import math
import numpy
import pylab
import sys
import timeit

from raleigh.interfaces.pca import pca
from raleigh.core.solver import Options
from raleigh.interfaces.truncated_svd import UserStoppingCriteria

try:
    input = raw_input
except NameError:
    pass


def _norm(a, axis):
    return numpy.apply_along_axis(numpy.linalg.norm, axis, a)


class Probe:
    '''An object of this class will monitor the computation if passed via
       UserStoppingCriteria constructor to Options.stopping_criteria.
    '''
    def __init__(self, images):
        self.images = images
        self.shape = images.shape
        self.img_list = None
        m = self.shape[0]
        ny = self.shape[1]
        nx = self.shape[2]
        self.nx = nx
        self.ny = ny
        # compute the norm of zero-average-shifted data samples
        n = nx*ny
        data = images.reshape((m, n))
        mean = numpy.mean(data, axis=0).reshape((1, n))
        s = numpy.linalg.norm(mean)
        b = numpy.dot(data, mean.T)
        t = _norm(data, axis=1).reshape((m, 1))
        ones = numpy.ones((m, 1), dtype=data.dtype)
        x = t*t - 2*b + s*s*ones
        self.nrms = numpy.sqrt(abs(x)).reshape((m,))

    def inspect(self, mean, sigma, left, right):
        '''This method will be called each time a bunch of singular vectors have
           converged.

           mean : mean image
           sigma: singular values computed so far
           left : left singular vectors computed so far
           right: right singular vectors computed so far

           This data can be used to judge whether sufficient number of
           eigenimages have been computed.
        '''
        # compute coordinates (reduced features data)
        s = numpy.reshape(sigma, (1, sigma.shape[0]))
        u = left*s
        # compute the errors
        proj = _norm(u, axis=1)
        nrms_sqr = self.nrms * self.nrms
        proj_sqr = proj*proj
        errs_sqr = nrms_sqr - proj_sqr
        err_fro = math.sqrt(numpy.sum(errs_sqr)/numpy.sum(nrms_sqr))
        i = sigma.shape[0] - 1
        msg = 'sigma[%d] = %.1e*sigma[0], truncation error %.1e'
        msg = msg % (i, sigma[i]/sigma[0], err_fro)
        # visualize the current approximation to images
        while True:
            ans = input(msg + ' h|q|s> ')
            vals = ans.split()
            nvals = len(vals)
            if nvals < 1:
                break
            if vals[0] == 'h':
                print('q               : to stop')
                print('s im1 [im2] ... : to show listed images')
                print('s               : to show previously selected images')
                print('any other input : to continue')
                continue
            if vals[0] == 'c':
                break
            if vals[0] == 'q':
                return True
            if vals[0] == 's':
                nimgs = nvals - 1
                if nimgs < 1:
                    imgs = self.img_list
                    if imgs is None:
                        print('usage: s img1 [img2] ...')
                        continue
                else:
                    imgs = vals[1:]
                nimgs = len(imgs)
                nim = left.shape[0]
                for i in range(nimgs):
                    im = int(imgs[i])
                    if im < 0 or im >= nim:
                        continue
                    image = self.images[im,:,:]
                    pylab.imshow(image, cmap='gray')
                    img = numpy.dot(u[im,:], right.T) + mean
                    pca_image = numpy.reshape(img, (ny, nx))
                    pylab.figure()
                    pylab.title('PCA approximation of the image')
                    pylab.imshow(pca_image, cmap='gray')
                    pylab.show()
                self.img_list = imgs
        return False
        

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

opt = Options()
probe = Probe(images)
opt.stopping_criteria = UserStoppingCriteria(images, probe=probe)

images = numpy.reshape(images, (m, n))

start = timeit.default_timer()
if err_tol > 0:
    mean, coord, eigim = pca(images, tol=err_tol, arch=arch, verb=1)
else:
    mean, coord, eigim = pca(images, opt=opt, arch=arch, verb=1)
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
