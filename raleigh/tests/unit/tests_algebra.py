'''Unit tests for vectors algebra.

Usage:
  tests_algebra [--help | options] <dim> <nv>

Arguments:
  dim  vector space dimension
  nv   number of vectors
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

n = int(args['<dim>'])
m = int(args['<nv>'])

import math
import numpy
import numpy.linalg as nla
import sys
import time

sys.path.append('../../..')

from raleigh.ndarray.numpy_algebra import Vectors as numpyVectors
from raleigh.ndarray.cblas_algebra import Vectors as cblasVectors
from raleigh.cuda.cublas_algebra import Vectors as cublasVectors
import raleigh.cuda.cuda as cuda

numpy.random.seed(1) # make results reproducible

def test(u, v):

    u_numpy = numpyVectors(u)
    v_numpy = numpyVectors(v)

    u_cblas = cblasVectors(u.copy())
    v_cblas = cblasVectors(v.copy())

    u_cublas = cublasVectors(u)
    v_cublas = cublasVectors(v)

    print('----\n testing numpy dot...')
    start = time.time()
    p = u_numpy.dot(v_numpy)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(p)
    print('time: %.2e' % elapsed)

    print('----\n testing cblas dot...')
    start = time.time()
    q = u_cblas.dot(v_cblas)
    stop = time.time()
    elapsed = stop - start
    print('error: %e' % (nla.norm(q - p)/s))
    print('time: %.2e' % elapsed)

    print('----\n testing cublas dot...')
    start = time.time()
    q = u_cublas.dot(v_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('error: %e' % (nla.norm(q - p)/s))
    print('time: %.2e' % elapsed)

    print('----\n testing numpy multiply...')
    start = time.time()
    u_numpy.multiply(p, v_numpy)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    s = numpy.sqrt(v_numpy.dots(v_numpy))

    print('----\n testing cblas multiply...')
    start = time.time()
    u_cblas.multiply(p, v_cblas)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = numpy.sqrt(v_cblas.dots(v_cblas))
    print('error: %e' % (nla.norm(t - s)/nla.norm(s)))

    print('----\n testing cublas multiply...')
    start = time.time()
    u_cublas.multiply(p, v_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = numpy.sqrt(v_cublas.dots(v_cublas))
    print('error: %e' % (nla.norm(t - s)/nla.norm(s)))

    print('----\n testing numpy add...')
    start = time.time()
    v_numpy.add(u_numpy, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    r = numpy.sqrt(v_numpy.dots(v_numpy))
    print('error: %e' % (nla.norm(r)/nla.norm(s)))

    print('----\n testing cblas add...')
    start = time.time()
    v_cblas.add(u_cblas, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    r = numpy.sqrt(v_cblas.dots(v_cblas))
    print('error: %e' % (nla.norm(r)/nla.norm(s)))

    print('----\n testing cublas add...')
    start = time.time()
    v_cublas.add(u_cublas, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    r = numpy.sqrt(v_cublas.dots(v_cublas))
    print('error: %e' % (nla.norm(r)/nla.norm(s)))

try:
    dt = numpy.float32
##    u = numpy.ones((m, n), dtype = dt)
##    v = numpy.ones((m, n), dtype = dt)
    u = numpy.random.randn(m, n).astype(dt)
    v = numpy.random.randn(m, n).astype(dt)

    test(u, v)
##    test(u + 1j*v, v - 2j*u)

except error as err:
    print('%s' % err.value)
