# -*- coding: utf-8 -*-
'''Computes PCs for a set of 2D images using truncated_svd and svds
until the error of PCA approximation falls below the tolerance for each image.

Usage:
  images [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   CG block size [default: -1]
  -e <err> , --imerr=<err>   image approximation error tolerance [default: 0.3]
  -n <nim> , --nimgs=<nim>   number of images to use (negative: all) 
                             [default: -1]
  -t <tol> , --restol=<tol>  residual tolerance [default: 1e-3]
  -f, --full  compute full SVD too (using scipy.linalg.svd)

Created on Wed Sep  5 14:44:23 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

#path = args['--path']
file = args['<data>']
ni = int(args['--nimgs'])
err_tol = float(args['--imerr'])
block_size = int(args['--bsize'])
#svec_tol = float(args['--svtol'])
tol = float(args['--restol'])
arch = args['--arch']
full = args['--full']

import numpy
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse.linalg import svds
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Options
from raleigh.svd import truncated_svd

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
    m = m_all
    images = all_images

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))

if block_size < 1:
    b = max(1, min(m, n)//100)
    block_size = 32
    while block_size <= b - 16:
        block_size += 32
    print('using block size %d' % block_size)

print('\n--- solving with raleigh.svd.truncated_svd...')
opt = Options()
opt.block_size = block_size
opt.max_iter = 300
opt.verbosity = -1
opt.convergence_criteria.set_error_tolerance \
    ('relative residual tolerance', tol)
start = time.time()
sigma, u, vt = truncated_svd(images, opt, tol = err_tol, arch = arch)
stop = time.time()
time_r = stop - start
ncon = sigma.shape[0]
print('\n%d singular vectors computed' % ncon)

dtype = numpy.float32

if full:
    print('\n--- solving with scipy.linalg.svd...')
    start = time.time()
#    s = sla.svd(images, full_matrices = False, compute_uv = False)
    u, s, vt = sla.svd(images, full_matrices = False)
    stop = time.time()
    time_f = stop - start
#    print(sigma[0], sigma[-1])
#    print(s[0], s[ncon - 1])
    print('\n full SVD time: %.1e' % time_f)

print('\n--- solving with restarted scipy.sparse.linalg.svds...')

sigma = numpy.ndarray((0,), dtype = dtype)
vt = numpy.ndarray((0, n), dtype = dtype)
norms = numpy.amax(nla.norm(images, axis = 1))

start = time.time()

while True:
    u, s, vti = svds(images, k = block_size, tol = tol)
    sigma = numpy.concatenate((sigma, s[::-1]))
    vt = numpy.concatenate((vt, vti[::-1, :]))
    print('last singular value computed: %e' % s[0])
    print('deflating...')
    images -= numpy.dot(u*s, vti)
    errs = numpy.amax(nla.norm(images, axis = 1))/norms
    print('max SVD error: %.3e' % errs)
    if errs <= err_tol or err_tol <= 0:
        break
    print('restarting...')

stop = time.time()
time_s = stop - start

print('\n%d singular vectors computed' % sigma.shape[0])

if err_tol > 0: # these timings make sense for non-interactive mode only
    print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))

print('\ndone')