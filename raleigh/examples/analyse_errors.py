# -*- coding: utf-8 -*-
'''Analyses errors of a partial SVD.

SVD data is read from files <prefix>sigma.npy, <prefix>eigim.npy and 
<prefix>coord.npy, which must be in the same folder as the images file.

Usage:
    analyse_errors [--help | -h | options] <file> <prefix>

Arguments:
    file       images file (must end with .npy)
    prefix     partial svd data prefix

Created on Wed Aug 22 11:13:17 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<file>']
pref = args['<prefix>']

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy
import numpy.linalg as nla
import pylab

l = file.rfind('/')
if l < 0:
    path = '.'
else:
    path = file[:l]
#print(path)

filename = path + '/%ssigma.npy' % pref
print('loading singular values from %s...' % filename)
sigma = numpy.load(filename)
#filename = path + '/%su.npy' % pref
filename = path + '/%seigim.npy' % pref
print('loading eigenimages from %s...' % filename)
u = numpy.load(filename)
#filename = path + '/%sv.npy' % pref
filename = path + '/%scoord.npy' % pref
print('loading images coordinates in eigenimages basis from %s...' % filename)
v = numpy.load(filename)

nsv, nyu, nxu = u.shape
m = v.shape[1]
print('%d eigenimages of size %dx%d loaded' % (nsv, nyu, nxu))

print('loading images from %s...' % file)
images = numpy.load(file)
ni, ny, nx = images.shape
if ni > m:
    ni = m
    images = images[:ni,:,:]
print('%d images of size %dx%d loaded' % (ni, ny, nx))
vmax = numpy.amax(images)

if ni != m or nx != nxu or ny != nyu:
    raise ValueError('data sizes (%d, %d, %d) and (%d, %d, %d) do not match' % \
          (ni, ny, nx, m, nyu, nxu))

pylab.figure()
pylab.plot(numpy.arange(1, nsv + 1, 1), sigma)
pylab.xscale('log')
pylab.yscale('log')
pylab.grid()
pylab.title('singular values')
pylab.show()

n = nx*ny
images = numpy.reshape(images, (ni, n))
u = numpy.reshape(u, (nsv, n))

print('measuring svd errors...')
norm = nla.norm(images, axis = 1)
proj = nla.norm(v, axis = 0)
err = numpy.sqrt(norm*norm - proj*proj)/norm
ind = numpy.argsort(-err)
pylab.figure()
pylab.plot(numpy.arange(1, ni + 1, 1), err[ind])
pylab.title('errors')
pylab.show()
#k = 100
#print('%d worst approximations:' % k)
#print('images:')
#print(ind[:k])
#print('errors:')
#print(err[ind[:k]])

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= ni:
        break
    image = numpy.reshape(images[i,:], (ny, nx))
    print('partial svd error: %.1e' % err[i])
    imgi = numpy.zeros((ny, nx), dtype = images.dtype)
    bitmaps = []
    fig = plt.figure()
    step = 1
    k = 0
    for j in range(nsv):
        imgi += numpy.reshape(u[j,:], (ny, nx))*v[j,i]
        x = (j*nx)//nsv
        imgi[:10, :x] = vmax/10
        if j % step == 0 or j == nsv - 1:
            bitmap = plt.imshow(imgi, cmap = 'gray')
            bitmaps.append([bitmap])
        k += 1
        if k == 32:
            if step < 1: #32:
                step *= 2
            k = 0
    ani = animation.ArtistAnimation \
        (fig, bitmaps, interval = 5, blit = True, repeat = False);
#    img = numpy.dot(u.T, v[:,i])
#    psvd_img = numpy.reshape(img, (ny, nx))
#    pylab.figure()
#    pylab.title('partial SVD approximation of the image')
#    pylab.imshow(psvd_img, cmap = 'gray')
#    pylab.figure()
#    pylab.plot(numpy.arange(1, nsv + 1, 1), v[:,i])
#    pylab.grid()
#    pylab.title('coordinates in eigenimage basis')
    pylab.show()
#    pylab.figure()
#    pylab.title('image %d' % i)
#    pylab.imshow(image, cmap = 'gray')
#    pylab.show()

print('done')