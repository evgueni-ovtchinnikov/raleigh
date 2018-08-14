# -*- coding: utf-8 -*-
"""Tests method Vectors.append().

Usage:
  tests_append [--help | options] <dim> <nv>

Arguments:
  dim  vector space dimension
  nv   number of vectors

Options:
  -d, --double
  -c, --complex

Created on Tue Aug 14 10:21:44 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

n = int(args['<dim>'])
m = int(args['<nv>'])
dble = args['--double']
cmplx = args['--complex']

import numpy
import numpy.linalg as nla
import sys
import time

sys.path.append('../../..')

from raleigh.ndarray.numpy_algebra import Vectors as numpyVectors
from raleigh.cuda.cublas_algebra import Vectors as cublasVectors
import raleigh.cuda.cuda as cuda

numpy.random.seed(1) # make results reproducible

def test(u, v):

    u_numpy = numpyVectors(u)
    v_numpy = numpyVectors(v)
    m = u_numpy.nvec()
    n = u_numpy.dimension()
    dt = u_numpy.data_type()
    w_numpy = numpyVectors(n, data_type = dt)

    u_cublas = cublasVectors(u)
    v_cublas = cublasVectors(v)
    w_cublas = cublasVectors(n, data_type = dt)
##    print(u_cublas.data())
##    print(w_cublas.data())

    print('----\n testing cublasVectors.append...')
    w_numpy.append(u_numpy)
    start = time.time()
    w_cublas.append(u_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(w_cublas.data() - w_numpy.data())
    print('error: %e' % t)
    print('time: %.2e' % elapsed)
    k = 4 #m//2
    l = m - k
    v_numpy.select(k, l)
    v_cublas.select(k, l)
    w_numpy.append(v_numpy)
    w_cublas.append(v_cublas)
    t = nla.norm(w_cublas.data() - w_numpy.data())
    print('error: %e' % t)
    k = 5
    l = m - k
    v_numpy.select(k, l)
    v_cublas.select(k, l)
    w_numpy.append(v_numpy)
    w_cublas.append(v_cublas)
    t = nla.norm(w_cublas.data() - w_numpy.data())
    print('error: %e' % t)
    v_numpy.select_all()
    v_cublas.select_all()

try:
    if dble:
        print('running in double precision...')
        dt = numpy.float64
    else:
        print('running in single precision...')
        dt = numpy.float32
##    u = numpy.ones((m, n), dtype = dt)
##    v = numpy.ones((m, n), dtype = dt)
    u = numpy.random.randn(m, n).astype(dt)
    v = numpy.random.randn(m, n).astype(dt)

    if cmplx:
        print('testing on complex data...')
        test(u + 1j*v, v - 2j*u)
    else:
        print('testing on real data...')
        test(u, v)

    print('done')

except Exception as e:
    print(e)
