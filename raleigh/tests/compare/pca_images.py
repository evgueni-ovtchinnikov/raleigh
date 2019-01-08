# -*- coding: utf-8 -*-
'''Computes PCs for a set of 2D images using raleigh.pca and scipy svd solvers.

Usage:
  pca_images [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   CG block size [default: -1]
  -e <err> , --imerr=<err>   image approximation error tolerance [default: 0.3]
  -n <nim> , --nimgs=<nim>   number of images to use (negative: all) 
                             [default: -1]
  -c <npc> , --npcs=<npc>    number of PCs to compute (negative: unknown)
                             [default: -1]
  -t <tol> , --restol=<tol>  residual tolerance [default: 1e-3]
  -f, --full  compute full SVD using scipy.linalg.svd
  -s, --svds  compute truncated SVD using scipy.sparse.linalg.svds

Created on Wed Sep  5 14:44:23 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<data>']
ni = int(args['--nimgs'])
npc = int(args['--npcs'])
err_tol = float(args['--imerr'])
block_size = int(args['--bsize'])
tol = float(args['--restol'])
arch = args['--arch']
run_svd = args['--full']
run_svds = args['--svds']

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
from raleigh.svd import pca

def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = numpy.linalg.norm(w, axis = 0)
    return s

numpy.random.seed(1) # make results reproducible

images = numpy.load(file)

m_all, ny, nx = images.shape
n = nx*ny

if ni < 0 or ni > m_all:
    ni = m_all

if ni < m_all:
    print('using first %d images only...' % ni)
    m = ni
    images = images[:m,:,:]
else:
    m = m_all
#    images = all_images

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))
dtype = images.dtype.type

##if block_size < 1:
##    b = max(1, min(m, n)//100)
##    block_size = 32
##    while block_size <= b - 16:
##        block_size += 32
##    print('using block size %d' % block_size)

print('\n--- solving with raleigh.svd.pca...')
opt = Options()
opt.block_size = block_size
#opt.max_iter = 1000
opt.verbosity = -1
opt.convergence_criteria.set_error_tolerance \
    ('relative residual tolerance', tol)
start = time.time()
sigma_r, u_r, vt_r = pca(images, opt, npc = npc, tol = err_tol, arch = arch)
stop = time.time()
time_r = stop - start
ncon = sigma_r.shape[0]
if err_tol > 0 or npc > 0: 
    print('\n%d singular vectors computed in %.1e sec' % (ncon, time_r))

if err_tol > 0:
    print('\n--- solving with restarted scipy.sparse.linalg.svds...')
else:
    print('\n--- solving with scipy.sparse.linalg.svds...')

sigma_s = numpy.ndarray((0,), dtype = dtype)
vt_s = numpy.ndarray((0, n), dtype = dtype)
norms = nla.norm(images, axis = 1)
vmin = numpy.amin(norms)
vmax = numpy.amax(norms)
print(vmin,vmax)

if run_svd:
    images0 = images.copy()

if npc <= 0:
    if block_size < 1:
        # use default pca block size
        b = max(1, min(m, n)//100)
        block_size = 32
        while block_size <= b - 16:
            block_size += 32
        print('using block size %d' % block_size)
    npc = block_size
else:
    err_tol = 0

start = time.time()

e = numpy.ones((n, 1), dtype = dtype)
s = numpy.dot(images, e)/n
images -= numpy.dot(s, e.T)
norms = nla.norm(images, axis = 1)
vmin = numpy.amin(norms)
vmax = numpy.amax(norms)
print(vmin,vmax)

#coord = numpy.reshape(sigma_r * u_r, (m, ncon))
#nrm = nla.norm(images, axis = 1)
#diff = images - numpy.dot(coord, vt_r)
#err = nla.norm(diff, axis = 1)/nrm
#print('pca error (raleigh): %.1e' % numpy.amax(err))

while run_svds:
    u, s, vti = svds(images, k = npc, tol = tol)
    sigma_s = numpy.concatenate((sigma_s, s[::-1]))
    vt_s = numpy.concatenate((vt_s, vti[::-1, :]))
    stop = time.time()
    time_s = stop - start
    sl = s[0]
    sl_rel = sl/sigma_s[0]
    print('%.2f sec: last singular value computed: %e = %.2e*sigma[0]' % \
        (time_s, sl, sl_rel))
    if err_tol <= 0:
        break
    print('deflating...')
    images -= numpy.dot(u*s, vti)
    errs = numpy.amax(nla.norm(images, axis = 1)/norms)
    print('max SVD error: %.3e' % errs)
    if errs <= err_tol:
        break
    print('restarting...')

stop = time.time()
time_s = stop - start
ncon = sigma_s.shape[0]
print('\n%d singular vectors computed in %.1e sec' % (ncon, time_s))

if run_svd:
    images = images0
    print('\n--- solving with scipy.linalg.svd...')
    start = time.time()
    e = numpy.ones((n, 1), dtype = dtype)
    s = numpy.dot(images, e)/n
    images -= numpy.dot(s, e.T)
    u0, sigma0, vt0 = sla.svd(images, full_matrices = False, overwrite_a = True)
    stop = time.time()
    time_f = stop - start
    print('\n full SVD time: %.1e' % time_f)
    n_r = min(sigma_r.shape[0], sigma0.shape[0])
    if n_r > 0:
        err_vec = vec_err(vt0.T[:,:n_r], vt_r.T[:,:n_r])
        err_val = abs(sigma_r[:n_r] - sigma0[:n_r])/sigma0[0]
        print('\nmax singular vector error (raleigh): %.1e' % numpy.amax(err_vec))
        print('\nmax singular value error (raleigh): %.1e' % numpy.amax(err_val))
    n_s = min(sigma_s.shape[0], sigma0.shape[0])
    if n_s > 0:
        err_vec = vec_err(vt0.T[:,:n_s], vt_s.T[:,:n_s])
        err_val = abs(sigma_s[:n_s] - sigma0[:n_s])/sigma0[0]
        print('\nmax singular vector error (svds): %.1e' % numpy.amax(err_vec))
        print('\nmax singular value error (svds): %.1e' % numpy.amax(err_val))
    #print(sigma0[:1000])

if err_tol > 0 or npc > 0: 
    # these timings make sense for non-interactive mode only
    print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))

print('\ndone')
