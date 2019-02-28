# -*- coding: utf-8 -*-
'''Computes PCs for a set of 2D images using raleigh.pca and scipy svd solvers.

Usage:
  pca_images [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -b <blk> , --bsize=<blk>   CG block size [default: -1]
  -e <err> , --imerr=<err>   image approximation error tolerance [default: 0]
  -n <nim> , --nimgs=<nim>   number of images to use (negative: all) 
                             [default: -1]
  -c <npc> , --npcs=<npc>    number of PCs to compute (negative: unknown)
                             [default: -1]
  -t <tol> , --restol=<tol>  residual tolerance [default: 1e-3]
  -f, --full  compute full SVD using scipy.linalg.svd
  -r, --ral   compute truncated SVD using raleigh.lra
  -s, --svds  compute truncated SVD using scipy.sparse.linalg.svds
  -k, --skl   compute truncated SVD using sklearn.decomposition.PCA

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
run_skl = args['--skl']
run_ral = args['--ral']

import numpy
import numpy.linalg as nla
import scipy.linalg as sla
from sklearn.decomposition import PCA
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Options
from raleigh.apps.lra import pca

def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = numpy.linalg.norm(w, axis = 0)
    return s

numpy.random.seed(1) # make results reproducible

all_images = numpy.load(file, mmap_mode = 'r')

m_all, ny, nx = all_images.shape
n = nx*ny

if ni < 0 or ni > m_all:
    ni = m_all

if ni < m_all:
    print('using the first %d out of %d images only...' % (ni, m_all))
    m = ni
else:
    m = m_all
    print('using all %d images...' % m)
images = all_images[:m,:,:].copy()

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))
dtype = images.dtype.type
e = numpy.ones((m, 1), dtype = dtype)
a = numpy.dot(e.T, images)/m

if run_ral:
    print('\n--- solving with raleigh.svd.pca...')
    opt = Options()
    opt.block_size = block_size
    opt.verbosity = -1
    opt.max_quota = 0.9
    if err_tol <= 0:
        nc = npc
    else:
        nc = -1
    start = time.time()
    mean, u_r, sigma_r, vt_r = pca(images, opt, npc = nc, tol = err_tol, \
        arch = arch)
    stop = time.time()
    time_r = stop - start
    ncon = sigma_r.shape[0]
    if err_tol > 0 or npc > 0: 
        print('\n%d singular vectors computed in %.1e sec' % (ncon, time_r))
    print('first and last singular values: %e %e' % (sigma_r[0], sigma_r[-1]))
    
    norms = nla.norm(images, axis = 1)
    imgs = numpy.dot(sigma_r*u_r, vt_r)
    mean = numpy.mean(imgs, axis = 0)
    print(numpy.amax(mean/a))
    imgs -= images
    imgs += numpy.dot(e, a)
    errs = nla.norm(imgs, axis = 1) #/norms
    print('max PCA error: %.1e' % (numpy.amax(errs)/numpy.amax(norms)))
else:
    sigma_r = numpy.ndarray((0,), dtype = dtype)

if npc <= 0:
    npc = max(1, min(m, n)//10)

if run_skl:
    if err_tol > 0:
        print('\n--- solving with restarted sklearn.decomposition.PCA...')
    else:
        print('\n--- solving with sklearn.decomposition.PCA...')
    images = all_images[:m,:,:].copy()
    images = numpy.reshape(images, (m, n))
    norms = nla.norm(images, axis = 1)
    start = time.time()
    skl_svd = PCA(npc)
    sigma_skl = numpy.ndarray((0,), dtype = dtype)
    vt_skl = numpy.ndarray((0, n), dtype = dtype)
    while run_skl:
        skl_svd.fit(images)
        s = skl_svd.singular_values_
        vti = skl_svd.components_
        sigma_skl = numpy.concatenate((sigma_skl, s))
        vt_skl = numpy.concatenate((vt_skl, vti))
        stop = time.time()
        time_s = stop - start
        pcs = vt_skl.shape[0]
        print('%.2f sec: last singular value computed: sigma[%d] = %e' \
            % (time_s, pcs - 1, s[-1]))
        if err_tol <= 0:
            break
        print('deflating...')
        imgs = images - numpy.dot(e, a)
        images -= numpy.dot(numpy.dot(imgs, vti.T), vti)
        imgs = images - numpy.dot(e, a)
        errs = numpy.amax(nla.norm(imgs, axis = 1))/numpy.amax(norms)
        print('max SVD error: %.3e' % errs)
        if errs <= err_tol:
            break
        print('restarting...')
    stop = time.time()
    time_k = stop - start
    print('sklearn time: %.1e' % time_k)
    imgs = images - numpy.dot(e, a)
    imgs -= numpy.dot(numpy.dot(imgs, vt_skl.T), vt_skl)
    errs = nla.norm(imgs, axis = 1) #/norms
    print('max PCA error: %.1e' % (numpy.amax(errs)/numpy.amax(norms)))

if run_svd:
    print('\n--- solving with scipy.linalg.svd...')
    images = all_images[:m,:,:].copy()
    images = numpy.reshape(images, (m, n))
    start = time.time()
    images -= numpy.dot(e, a)
    u0, sigma0, vt0 = sla.svd(images, full_matrices = False, overwrite_a = True)
    stop = time.time()
    time_f = stop - start
    print('\n full SVD time: %.1e' % time_f)
    print(sigma0[0])
    if npc > 0:
        print(sigma0[npc - 1])
    n_r = min(sigma_r.shape[0], sigma0.shape[0])
    if n_r > 0:
        err_vec = vec_err(vt0.T[:,:n_r], vt_r.T[:,:n_r])
        err_val = abs(sigma_r[:n_r] - sigma0[:n_r])/sigma0[0]
        print('\nmax singular vector error (raleigh): %.1e' % numpy.amax(err_vec))
        print('\nmax singular value error (raleigh): %.1e' % numpy.amax(err_val))
    n_k = min(sigma_skl.shape[0], sigma0.shape[0])
    if n_k > 0:
        err_vec = vec_err(vt0.T[:,:n_k], vt_skl.T[:,:n_k])
        err_val = abs(sigma_skl[:n_k] - sigma0[:n_k])/sigma0[0]
        print('\nmax singular vector error (sklearn): %.1e' % numpy.amax(err_vec))
        print('\nmax singular value error (sklearn): %.1e' % numpy.amax(err_val))

if run_ral and (err_tol > 0 or npc > 0):
    # makes sense for non-interactive mode only
    print('\n raleigh time: %.1e' % time_r)
if run_skl:
    print('\n sklearn time: %.1e' % time_k)

print('\ndone')