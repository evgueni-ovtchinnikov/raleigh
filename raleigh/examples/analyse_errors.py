# -*- coding: utf-8 -*-
'''Analyses errors of a partial SVD.

SVD data is read from files <prefix>sigma.npy, <prefix>eigim.npy and 
<prefix>coord.npy, which must be in the same folder as the images file.

Usage:
    analyse_errors [--help | -h | options] <file>

Arguments:
    file       images file (must end with .npy)

Created on Wed Aug 22 11:13:17 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<file>']

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy
import numpy.linalg as nla
import pylab

l = file.rfind('/')
if l < 0:
    path = './'
else:
    path = file[: l + 1]
path = './'

filename = path + 'eigenimages.npy'
print('loading eigenimages from %s...' % filename)
u = numpy.load(filename)
filename = path + 'coordinates.npy'
print('loading images coordinates in eigenimages basis from %s...' % filename)
v = numpy.load(filename)
filename = path + 'mean.npy'
try:
    mean = numpy.load(filename)
    print('loaded mean image from %s...' % filename)
except:
    mean = None

nc, nyu, nxu = u.shape
m = v.shape[0]
print('%d eigenimages of size %dx%d loaded' % (nc, nyu, nxu))

print('loading images from %s...' % file)
images = numpy.load(file)
ni, ny, nx = images.shape
print('%d images of size %dx%d loaded' % (ni, ny, nx))
vmax = numpy.amax(images)

if nx != nxu or ny != nyu:
    raise ValueError('data sizes (%d, %d, %d) and (%d, %d, %d) do not match' % \
          (ni, ny, nx, m, nyu, nxu))

sigma = numpy.linalg.norm(v, axis = 0)

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

print('measuring lra errors...')
norm0 = nla.norm(images[:m, :], axis = 1)
if mean is None:
    #norm = norm0
    pca_images = numpy.dot(v, u)
else:
    ones = numpy.ones((m, 1))
    mean = numpy.reshape(mean, (1, n))
    #norm = nla.norm(images[:m, :] - numpy.dot(ones, mean), axis = 1)
    pca_images = numpy.dot(v, u) + numpy.dot(ones, mean)
mean = numpy.reshape(mean, n)
proj = nla.norm(v, axis = 1)
err = nla.norm(images[:m, :] - pca_images, axis = 1) #/norm0
#err1 = numpy.sqrt(abs(norm*norm - proj*proj))/norm0
ind = numpy.argsort(-err)
pylab.figure()
pylab.plot(numpy.arange(1, m + 1, 1), err[ind])
pylab.title('errors')
pylab.show()
k = 50
print('%d worst approximations:' % k)
print('images:')
print(ind[:k])
##print('errors:')
##print(err[ind[:k]])

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= ni:
        break
    image = numpy.reshape(images[i,:], (ny, nx))
    print('partial svd error: %.1e' % err[i])
    if i < m:
        w = v[i, :]
    else:
        w = numpy.dot(u, images[i, :])
    pylab.figure()
    pylab.title('image %d' % i)
    img = images[i, :]
    image = numpy.reshape(img, (ny, nx))
    pylab.imshow(image, cmap = 'gray')
    img = numpy.dot(w, u)
    if mean is not None:
        img += mean
    pca_image = numpy.reshape(img, (ny, nx))
    pylab.figure()
    pylab.title('PCA approximation of the image')
    pylab.imshow(pca_image, cmap = 'gray')
    pylab.show()

print('done')
