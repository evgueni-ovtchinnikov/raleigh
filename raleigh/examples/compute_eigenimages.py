# -*- coding: utf-8 -*-
'''
Incremental PCA demo. Computes Principal Components for a set of 2D images
in small portions until stopped by user's entering 'n' in answer to 'more?'
or the error of PCA approximation falls below the tolerance for each image

Usage:
  compute_eigenimages [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -n <nim> , --nimgs=<nim>   number of images to use (< 0: all) [default: -1]
  -m <mim> , --mimgs=<mim>   number of images to add for update [default: 0]
  -e <err> , --imerr=<err>   image approximation error tolerance (<= 0: not set,
                             run in interactive mode) [default: 0]
  -b <blk> , --bsize=<blk>   CG block size (< 0: auto) [default: -1]
  -t <tol> , --rtol=<tol>    residual tolerance [default: 1e-3]
  -a <arch>, --arch=<arch>   architecture [default: cpu]

Created on Mon Jun 18 12:10:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

#path = args['--path']
file = args['<data>']
ni = int(args['--nimgs'])
mi = int(args['--mimgs'])
err_tol = float(args['--imerr'])
block_size = int(args['--bsize'])
tol = float(args['--rtol'])
arch = args['--arch']

import numpy
import numpy.linalg as nla
import sys
import time

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Options
from raleigh.apps.lra import pca
#from raleigh.svd import pca

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

if ni + mi > m_all:
    mi = m_all - ni

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))
#ones = numpy.ones((m, 1), dtype = images.dtype.type)
#mean = numpy.dot(ones.T, images)/m

if block_size < 1:
    b = max(1, min(m, n)//100)
    block_size = 32
    while block_size <= b - 16:
        block_size += 32
    print('using block size %d' % block_size)

opt = Options()
opt.block_size = block_size
opt.max_iter = 500
opt.verbosity = -1
#opt.convergence_criteria.set_error_tolerance \
#    ('residual tolerance', tol)
#    ('kinematic eigenvector error', svec_tol)

start = time.time()
mean, u, sigma, vt = pca(images, opt, tol = err_tol, arch = arch)
#sigma, u, vt = truncated_svd(images, opt, tol = err_tol, arch = arch, shift = True)
stop = time.time()
elapsed_time = stop - start

ncon = sigma.shape[0]
print('%d eigenimages computed' % ncon)
if err_tol > 0:
    print('elapsed time: %.2e' % elapsed_time)

if mi > 0:
    print('adding %d images...' % mi)
    m = ni + mi
    images = all_images[:m,:,:]
    images = numpy.reshape(images, (m, n))
    #opt = Options()
    opt.max_iter = 500
    opt.verbosity = -1
#    opt.convergence_criteria.set_error_tolerance \
#        ('residual eigenvector error', svec_tol)
    opt.block_size = ncon
    start = time.time()
    sigma, u, vt = pca \
        (images, opt, npc = ncon + mi, tol = err_tol, ipc = vt.T, arch = arch)
#    sigma, u, vt = truncated_svd \
#        (images, opt, nsv = ncon + mi, tol = err_tol, isv = vt.T, shift = True, arch = arch)
##        (images, opt, tol = err_tol, isv = vt.T, shift = True, arch = arch)
    stop = time.time()
    elapsed_time = stop - start
    ncon = sigma.shape[0]
    print('%d eigenimages computed' % ncon)
    if err_tol > 0:
        print('elapsed time %.2e sec' % elapsed_time)

dt = images.dtype.type
#e = numpy.ones((n, 1), dtype = dt)
#s = numpy.dot(images, e)/n
#images -= numpy.dot(s, e.T)
nrm = nla.norm(images, axis = 1)
coord = numpy.reshape((sigma*u).T, (ncon, m))
#diff = images - numpy.dot(ones, mean) 
#diff -= numpy.dot(coord.T, vt)
#err = nla.norm(diff, axis = 1)/nrm
#print('svd error %e' % numpy.amax(err))

import pylab

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= m:
        break
    pylab.figure()
    pylab.title('image %d' % i)
    img = images[i,:]
    image = numpy.reshape(img, (ny, nx))
    pylab.imshow(image, cmap = 'gray')
    img = numpy.dot(coord[:,i].T, vt) + mean
    pca_image = numpy.reshape(img, (ny, nx))
    pylab.figure()
    pylab.title('PCA approximation of the image')
    pylab.imshow(pca_image, cmap = 'gray')
    pylab.show()

print('saving...')
mean = numpy.reshape(mean, (ny, nx))
eigim = numpy.reshape(vt, (ncon, ny, nx))
numpy.save('mean.npy', mean)
numpy.save('eigenimages.npy', eigim)
numpy.save('coordinates.npy', coord)
numpy.save('sigma.npy', sigma[:ncon])

print('done')