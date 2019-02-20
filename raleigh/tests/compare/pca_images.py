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

import numpy
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA #, TruncatedSVD
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Options
#from raleigh.svd import pca
from raleigh.apps.lra import pca
#from raleigh.algebra import Vectors

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
    print('using all %d images...' % m)

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))
dtype = images.dtype.type
e = numpy.ones((m, 1), dtype = dtype)
a = numpy.dot(e.T, images)/m

print('\n--- solving with raleigh.svd.pca...')
opt = Options()
opt.block_size = block_size
#opt.max_iter = 1000
opt.verbosity = -1
opt.max_quota = 0.9
start = time.time()
mean, u_r, sigma_r, vt_r = pca(images, opt, npc = npc, tol = err_tol, \
    arch = arch)
stop = time.time()
time_r = stop - start
ncon = sigma_r.shape[0]
if err_tol > 0 or npc > 0: 
    print('\n%d singular vectors computed in %.1e sec' % (ncon, time_r))
print('first and last singular values: %e %e' % (sigma_r[0], sigma_r[-1]))

norms = nla.norm(images, axis = 1)
pc_images = numpy.dot(sigma_r*u_r, vt_r)
mean = numpy.mean(pc_images, axis = 0)
print(numpy.amax(mean/a))
diff = images - numpy.dot(e, a) - pc_images
errs = nla.norm(diff, axis = 1)/norms
print('max PCA error: %.1e' % numpy.amax(errs))

if run_svd or run_skl:
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

if run_skl:
    if err_tol > 0:
        print('\n--- solving with restarted sklearn.decomposition.PCA...')
    else:
        print('\n--- solving with sklearn.decomposition.PCA...')

norms = nla.norm(images, axis = 1)

start = time.time()
skl_svd = PCA(npc, tol = tol)
sigma_skl = numpy.ndarray((0,), dtype = dtype)
vt_skl = numpy.ndarray((0, n), dtype = dtype)
while run_skl:
    skl_svd.fit(images)
    s = skl_svd.singular_values_
    #print(s)
    vti = skl_svd.components_
    #print(vti.shape)
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
    errs = numpy.amax(nla.norm(imgs, axis = 1)/norms)
    print('max SVD error: %.3e' % errs)
    if errs <= err_tol:
        break
    print('restarting...')

stop = time.time()
time_k = stop - start
if run_skl:
    print('sklearn time: %.1e' % time_k)
    diff = images - numpy.dot(e, a)
    diff -= numpy.dot(numpy.dot(diff, vt_skl.T), vt_skl)
    errs = nla.norm(diff, axis = 1)/norms
    print('max PCA error: %.1e' % numpy.amax(errs))
    pcs = vt_skl.shape[0]
    k = min(ncon, pcs) - 1
    if k >= 0:
        print(sigma_r[k], sigma_skl[k])

if run_svd or run_skl:
    images = images0.copy()

if run_svds:
    if err_tol > 0:
        print('\n--- solving with restarted scipy.sparse.linalg.svds...')
    else:
        print('\n--- solving with scipy.sparse.linalg.svds...')

sigma_s = numpy.ndarray((0,), dtype = dtype)
vt_s = numpy.ndarray((0, n), dtype = dtype)
start = time.time()

norms = nla.norm(images, axis = 1)
images -= numpy.dot(e, a)

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
if run_svds:
    print('\n%d singular vectors computed in %.1e sec' % (ncon, time_s))

if run_svd:
    images = images0
    print('\n--- solving with scipy.linalg.svd...')
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
    n_s = min(sigma_s.shape[0], sigma0.shape[0])
    if n_s > 0:
        err_vec = vec_err(vt0.T[:,:n_s], vt_s.T[:,:n_s])
        err_val = abs(sigma_s[:n_s] - sigma0[:n_s])/sigma0[0]
        print('\nmax singular vector error (svds): %.1e' % numpy.amax(err_vec))
        print('\nmax singular value error (svds): %.1e' % numpy.amax(err_val))
    #print(sigma0[:1000])

if err_tol > 0 or npc > 0: 
    # these timings make sense for non-interactive mode only
    #print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))
    print('\n raleigh time: %.1e' % time_r)
    if run_skl:
        print('\n sklearn time: %.1e' % time_k)
    if run_svds:
        print('\n scipy svds time: %.1e' % time_s)

print('\ndone')