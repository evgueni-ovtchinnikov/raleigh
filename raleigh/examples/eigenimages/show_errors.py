# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''Compares images with their PCA approximations computed by the script
   compute_eigenimages.py.

Usage:
    show_errors [--help | -h] <images> <eigenimages>

Arguments:
    images       images file (must end with .npy)
    eigenimages  eigenimages file (must end with .npz)
'''

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import math
import numpy
import numpy.linalg as nla
import pylab
import sys


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


if have_docopt:
    __version__ = '0.1.0'
    args = docopt(__doc__, version=__version__)
    img_file = args['<images>']
    eigim_file = args['<eigenimages>']
else:
    narg = len(sys.argv)
    if narg < 3:
        print('Usage: show_errors <images_file> <eigenimages_file>')
    img_file = sys.argv[1]
    eigim_file = sys.argv[2]

eigim = numpy.load(eigim_file)
u = eigim['eigim']
v = eigim['coord']
mean = eigim['mean']
nc, nyu, nxu = u.shape
m = v.shape[0]
print('%d eigenimages of size %dx%d loaded' % (nc, nyu, nxu))

print('loading images from %s...' % img_file)
images = numpy.load(img_file)
ni, ny, nx = images.shape
print('%d images of size %dx%d loaded' % (ni, ny, nx))
vmax = numpy.amax(images)

if m > ni:
    m = ni
if nx != nxu or ny != nyu:
    raise ValueError('mismatching image sizes (%d, %d) and (%d, %d)' % \
          (ny, nx, nyu, nxu))

sigma = _norm(v, axis=0)

pylab.figure()
pylab.plot(numpy.arange(1, nc + 1, 1), sigma)
pylab.xscale('log')
pylab.yscale('log')
pylab.grid()
pylab.title('singular values')
pylab.show()

n = nx*ny
images = numpy.reshape(images, (ni, n))
u = numpy.reshape(u, (nc, n))

print('measuring PCA errors...')
norm = _norm(images[:m, :], axis=1)
ones = numpy.ones((m, 1))
mean = numpy.reshape(mean, (1, n))
pca_images = numpy.dot(v, u) + numpy.dot(ones, mean)
mean = numpy.reshape(mean, n)
norm = vmax*math.sqrt(n)
err = _norm(images[:m, :] - pca_images, axis=1)/norm
ind = numpy.argsort(-err)
pylab.figure()
pylab.plot(numpy.arange(1, m + 1, 1), err[ind])
pylab.title('PCA errors')
pylab.show()
k = 100
print('%d images with the largest PCA error:' % k)
print(ind[:k])

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= ni:
        break
    image = numpy.reshape(images[i,:], (ny, nx))
    print('PCA error: %.1e' % err[i])
    if i < m:
        w = v[i, :]
    else:
        w = numpy.dot(u, images[i, :])
    pylab.figure()
    pylab.title('image %d' % i)
    img = images[i, :]
    image = numpy.reshape(img, (ny, nx))
    pylab.imshow(image, cmap='gray')
    img = numpy.dot(w, u)
    if mean is not None:
        img += mean
    pca_image = numpy.reshape(img, (ny, nx))
    pylab.figure()
    pylab.title('PCA approximation of the image')
    pylab.imshow(pca_image, cmap='gray')
    pylab.show()

print('done')
