# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''Interactive Principal Component Analysis of 2D images.

Computes Principal Components for a set of 2D images interactively until
stopped by user.

--- A note on Linear Algebra style terminology adopted here ---
Due to the specific nature of this PCA application, principal components are
referred to as 'eigenimages'. Each image from the original data set is
approximated by a linear combination of eigenimages, hence the coefficients
of this linear combination are referred to as 'coordinates' (of the approximate
image in the orthonormal basis of eigenimages) - in some other PCA software
these are referred to as 'reduced features data'.

The computed eigenimages and coordinates and the mean image are saved in the
file eigenimages.npz as three variables named eigim, coord and mean.

If you have docopt:
-------------------
Usage:
  icompute_eigenimages [--help | -h | options] <images>

Arguments:
  images  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -n <nim> , --nimgs=<nim>   number of images to use (< 0: all) [default: -1]
  -a <arch>, --arch=<arch>   architecture [default: cpu]

If you do not have docopt:
--------------------------
  icompute_eigenimages <images> <pca_err> [gpu]
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
        self.tol = 0 # start in interactive mode
        self.first = True # to give help hint at the start
        self.img_list = None # no images to show yet
        m = self.shape[0]
        ny = self.shape[1]
        nx = self.shape[2]
        self.nx = nx
        self.ny = ny
        # compute the norms of centered (shifted-to-zero-average) data samples
        # to be used for the relative truncation error computation
        n = nx*ny
        data = images.reshape((m, n))
        mean = numpy.mean(data, axis=0).reshape((1, n))
        ones = numpy.ones((m, 1), dtype=data.dtype)
        r = numpy.linalg.norm(mean)
        s = numpy.dot(data, mean.T)
        t = _norm(data, axis=1).reshape((m, 1))
        nrms_sqr = t*t - 2*s + r*r*ones
        self.nrms = numpy.sqrt(abs(nrms_sqr)).reshape((m,))

    def inspect(self, mean, sigma, left, right):
        '''Principal components computed by RALEIGH's method pca are the right
           singular vectors of the centered (shifted to zero mean) data matrix
           of shape samples x features, and coordinates of i-th data sample
           approximation are i-th components of the left singular vectors
           multiplied by the respective singular values.

           The method 'inspect' will be called by the RALEIGH's core eigensolver
           each time a bunch of singular vectors have converged.

           Parameters:
           -----------
           mean : 1D numpy array
                  mean row of the data matrix
           sigma: 1D numpy array
                  singular values of the data matrix computed so far
           left : 2D numpy array
                  columns are left singular vectors of the data matrix computed
                  so far
           right: 2D numpy array
                  columns are right singular vectors of the data matrix computed
                  so far

           In the implementation below, these data are, on user's request,
           used to visualize PCA approximations to images, helping the user to
           decide whether a sufficient number of eigenimages have been computed.
        '''
        # compute coordinates (reduced features data)
        s = numpy.reshape(sigma, (1, sigma.shape[0]))
        u = left*s
        # compute the errors
        proj = _norm(u, axis=1)
        nrms_sqr = self.nrms * self.nrms
        proj_sqr = proj*proj
        errs_sqr = nrms_sqr - proj_sqr # Pythagorean theorem
        err_fro = math.sqrt(numpy.sum(errs_sqr)/numpy.sum(nrms_sqr))
        i = sigma.shape[0] - 1 # latest singular value
        msg = 'sigma[%d] = %.1e*sigma[0], truncation error %.1e'
        msg = msg % (i, sigma[i]/sigma[0], err_fro)
        if self.tol > 0: # we are in non-intractive mode
            print(msg)
            if err_fro < self.tol:
                self.tol = 0 # back to interactive mode
            else:
                return False # more eigenimages needed
        while True: # process user's choices
            if self.first:
                print('answer h to the prompt below to get usage help')
                self.first = False
            ans = input(msg + ' h|q|s|t> ')
            vals = ans.split()
            nvals = len(vals)
            if nvals < 1:
                break
            if vals[0] == 'h':
                print('q               : to stop the computation')
                print('s im1 [im2 ...] : to show listed images')
                print('s               : to show previously selected images')
                print('t tolerance     : to continue in non-iteractive mode')
                print('                  until the truncation error falls')
                print('                  below tolerance')
                print('any other input : to compute few more eigenimages')
                continue
            if vals[0] == 'q':
                return True # we are done
            if vals[0] == 's':
                # visualize the current approximation to images
                nimgs = nvals - 1
                if nimgs < 1:
                    imgs = self.img_list
                    if imgs is None:
                        print('usage: s im1 [im2 ...]')
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
                    pylab.figure()
                    pylab.title('image %d' % im)
                    pylab.imshow(image, cmap='gray')
                    img = numpy.dot(u[im, :], right.T) + mean
                    pca_image = numpy.reshape(img, (ny, nx))
                    pylab.figure()
                    pylab.title('PCA approximation of the image %d' % im)
                    pylab.imshow(pca_image, cmap='gray')
                    pylab.show()
                self.img_list = imgs
                continue
            if vals[0] == 't':
                self.tol = float(vals[1])
            break
        return False # more eigenimages needed
        

if have_docopt:
    __version__ = '0.1.0'
    args = docopt(__doc__, version=__version__)
    file = args['<images>']
    ni = int(args['--nimgs'])
    arch = args['--arch']
else:
    narg = len(sys.argv)
    if narg < 2:
        print('Usage: icompute_eigenimages <images_file> [gpu]')
        exit()
    file = sys.argv[1]
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

# instruct the RALEIGH's core eigensolver to call the above implementation
# of Probe.inspect() each time a bunch of singular vectors have converged.
opt = Options()
probe = Probe(images)
opt.stopping_criteria = UserStoppingCriteria(images, shift=True, probe=probe)

images = numpy.reshape(images, (m, n))

mean, coord, eigim = pca(images, opt=opt, arch=arch, verb=1)

ncon = eigim.shape[0]
print('%d eigenimages computed' % ncon)

print('saving...')
mean = numpy.reshape(mean, (ny, nx))
eigim = numpy.reshape(eigim, (ncon, ny, nx))
numpy.savez('eigenimages', eigim=eigim, coord=coord, mean=mean)

print('done')
