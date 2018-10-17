# -*- coding: utf-8 -*-
'''Compares PCA approximation with equivalent low resolution.

Images read from <images>.

PCs read from the file <prefix>eigim.npy and coordinates
from the file <prefix>coord.npy;
both must be in the same folder as the images file.

Usage:
    cmp_approx [--help | -h | options] <images> <prefix>

Arguments:
    images   images file
    prefix   PCs and coordinates prefix

Options:
  -l, --low_res_eqv  show the equivalent low resolution approximation

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<images>']
pref = args['<prefix>']
low = args['--low_res_eqv']

import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy
import numpy.linalg as nla
import pylab

def lower_res(image, r):
    ny, nx = image.shape
    my = (ny - 1)//r + 1
    mx = (nx - 1)//r + 1
    img = numpy.ndarray((ny, nx), dtype = image.dtype)
    y1 = 0
    for yy in range(my):
        y2 = min(ny, y1 + r)
        x1 = 0
        for xx in range(mx):
            x2 = min(nx, x1 + r)
            k = 0
            v = 0
            for y in range(y1, y2):
                for x in range(x1, x2):
                    k += 1
                    v += image[y, x]
            v /= k
            for y in range(y1, y2):
                for x in range(x1, x2):
                    img[y, x] = v
            if x2 < nx:
                x1 = x2
            else:
                break
        if y2 < ny:
            y1 = y2
        else:
            break
    return img                    

l = file.rfind('/')
if l < 0:
    path = '.'
else:
    path = file[:l]
#print(path)

filename = path + '/%ssigma.npy' % pref
print('loading singular values from %s...' % filename)
sigma = numpy.load(filename)
filename = path + '/%seigim.npy' % pref
print('loading eigenimages from %s...' % filename)
u = numpy.load(filename)
filename = path + '/%scoord.npy' % pref
print('loading images coordinates in eigenimages basis from %s...' % filename)
v = numpy.load(filename)

nsv, nyu, nxu = u.shape
m = v.shape[1]
print('%d eigenimages of size %dx%d loaded' % (nsv, nyu, nxu))

print('loading images from %s...' % file)
images = numpy.load(file)
ni, ny, nx = images.shape

##images_a = -numpy.sum(images, axis = 0)/ni
##pylab.figure()
##pylab.title('average image')
##pylab.imshow(images_a[:,:], cmap = 'gray')
##pylab.show()

##if ni > m:
##    ni = m
##    images = images[:ni,:,:]
print('%d images of size %dx%d loaded' % (ni, ny, nx))
vmax = numpy.amax(images)

##if ni != m or nx != nxu or ny != nyu:
if nx != nxu or ny != nyu:
    raise ValueError('data sizes (%d, %d, %d) and (%d, %d, %d) do not match' % \
          (ni, ny, nx, m, nyu, nxu))

n = nx*ny
u = numpy.reshape(u, (nsv, n))

if ni > m:
    mi = ni - m
    extra = images[m : ni, :, :]
    extra = numpy.reshape(extra, (mi, n))
    w = numpy.dot(u, extra.T)
    norm = nla.norm(extra, axis = 1)
    proj = nla.norm(w, axis = 0)
    err = numpy.sqrt(norm*norm - proj*proj)/norm
    ind = numpy.argsort(-err)
    k = min(mi, 20)
    print(ind[:k])
    print(err[ind[:k]])

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= ni:
        break
    pylab.figure()
    pylab.title('image %d' % i)
    pylab.imshow(images[i,:,:], cmap = 'gray')
    if i < m:
        img = numpy.dot(u.T, v[:,i])
    else:
        print('PC coordinates not available, projecting on PCs...')
        image = numpy.reshape(images[i,:,:], (n,))
        w = numpy.dot(u, image)
        img = numpy.dot(u.T, w)
    pca_img = numpy.reshape(img, (ny, nx))
    pylab.figure()
    pylab.title('PCA approximation of the image')
    pylab.imshow(pca_img, cmap = 'gray')
    if low:
        r = int(math.sqrt(n/nsv))
        mx = int(math.ceil(nx/r))
        my = int(math.ceil(ny/r))
        pylab.figure()
        pylab.title('equivalent %d-by-%d approximation' % (mx, my))
        low_img = lower_res(images[i,:,:], 6)
        pylab.imshow(low_img, cmap = 'gray')
    pylab.show()

##images = numpy.reshape(images, (ni, n))
##print(sigma[0])
##norms = (1, numpy.inf, 'fro')
##for nrm in norms:
##    print(numpy.linalg.norm(images, nrm))
##print(numpy.linalg.norm(images, 1))
##print(numpy.linalg.norm(images, numpy.inf))
##print(numpy.linalg.norm(images, 'fro'))
##cov_mx = abs(numpy.dot(images[1230:1259,:], images[1230:1259,:].T))
##print(numpy.amin(cov_mx))
##for i in range(cov_mx.shape[0]):
##    cov_mx[i,i] = 0
##print(numpy.amax(cov_mx))
##print(numpy.mean(abs(cov_mx)))

print('done')
