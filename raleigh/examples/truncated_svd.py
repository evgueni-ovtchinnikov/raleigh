# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Truncated SVD demo.

Usage:
  truncated_svd [--help | -h | options] <data>

Arguments:
  data  numpy .npy file containing the matrix.

Options:
  -a <arch>, --arch=<arch>   architecture [default: cpu]
  -d <svd> , --svd=<svd>     .npz file contaning singular values and vectors
                             of the data matrix (if available)
  -n <rank>, --rank=<rank>   truncated svd rank (negative: unknown a priori, 
                             thsh > 0 will be used as a criterion,
                             if thsh == 0, will run in interactive mode)
                             [default: -1]
  -s <thsh>, --thres=<thsh>  singular values threshold [default: 0.01]
  -t <vtol>, --vtol=<vtol>   singular vectors error tolerance [default: 1e-3]
  -v <verb>, --verb=<verb>   verbosity level [default: 0]
  -P, --pca                  compute also PCA
"""

try:
    from docopt import docopt
    __version__ = '0.1.0'
    have_docopt = True
except:
    have_docopt = False

import numpy
import numpy.linalg as nla
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 2

from raleigh.core.solver import Options
from raleigh.interfaces.truncated_svd import truncated_svd
from raleigh.interfaces.pca import pca


def norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = norm(w, axis = 0)
    return s


if have_docopt:
    args = docopt(__doc__, version=__version__)
    filename = args['<data>']
    A = numpy.load(filename)
    arch = args['--arch']
    rank = int(args['--rank'])
    th = float(args['--thres'])
    tol = float(args['--vtol'])
    verb = int(args['--verb'])
    do_pca = args['--pca']
    svd_data = None
    filename = args['--svd']
    if filename is not None:
        svd_data = numpy.load(filename)
    else:
        svd_data = None
else:
    narg = len(sys.argv)
    if narg < 2 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print('\nUsage:\n')
        print('python truncated_svd.py <data>')
        exit()
    data = sys.argv[1]
    print('\n=== docopt not found, using default options...\n')
    arch = 'cpu'
    rank = -1
    th = 0.01
    tol = 1e-3
    verb = 0
    do_pca = True
    svd_data = None

numpy.random.seed(1) # make results reproducible

m = A.shape[0]
if len(A.shape) > 2:
    n = numpy.prod(A.shape[1:])
    A = numpy.reshape(A, (m, n))
else:
    n = A.shape[1]
dtype = A.dtype.type

print('\n--- solving with truncated_svd...\n')
start = time.time()
u, sigma, vt = truncated_svd(A, nsv=rank, tol=th, vtol=tol, arch=arch)
stop = time.time()
time_tsvd = stop - start
print('\ntruncated svd time: %.1e' % time_tsvd)

nsv = sigma.shape[0]
print('\n%d singular vectors computed' % nsv)
nsv = numpy.sum(sigma > th)
print('%d singular values above threshold' % nsv)
if nsv > 0 and svd_data is not None:
    sigma0 = svd_data['sigma']
    v0 = svd_data['right']
    nsv = min(nsv, sigma0.shape[0])
    err_vec = vec_err(v0[:,:nsv], vt.transpose()[:,:nsv])
    err_val = abs(sigma[:nsv] - sigma0[:nsv])/sigma0[0]
    print('\nmax singular vector error: %.1e' % numpy.amax(err_vec))
    print('max singular value error: %.1e' % numpy.amax(err_val))
D = A - numpy.dot(sigma[:nsv]*u[:, :nsv], vt[:nsv, :])
err = numpy.amax(norm(D, axis=1))/numpy.amax(norm(A, axis=1))
print('\ntruncation error %.1e' % err)

print('\ndone')
