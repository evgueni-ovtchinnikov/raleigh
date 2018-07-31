'''Unit tests for vectors algebra.

Usage:
  tests_algebra [--help | options] <dim> <nv>

Arguments:
  dim  vector space dimension
  nv   number of vectors

Options:
  -d, --double
  -c, --complex
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

n = int(args['<dim>'])
m = int(args['<nv>'])
dble = args['--double']
cmplx = args['--complex']

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

    print('----\n testing numpy copy...')
    start = time.time()
    u_numpy.copy(v_numpy)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(v_numpy.data())
    print('time: %.2e' % elapsed)

    print('----\n testing cblas copy...')
    start = time.time()
    u_cblas.copy(v_cblas)
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(v_cblas.data() - v_numpy.data())/s
    print('error: %e' % t)
    print('time: %.2e' % elapsed)

    print('----\n testing cublas copy...')
    start = time.time()
    u_cublas.copy(v_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(v_cublas.data() - v_numpy.data())/s
    print('error: %e' % t)
    print('time: %.2e' % elapsed)

    m = u_numpy.nvec()
    ind = numpy.arange(m)
    for i in range(m - 1):
        ind[i] = ind[i + 1]
    ind[m - 1] = 0

    print('----\n testing numpy indexed copy...')
    start = time.time()
    u_numpy.copy(v_numpy, ind)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(v_numpy.data())
    print('time: %.2e' % elapsed)

    print('----\n testing cblas indexed copy...')
    start = time.time()
    u_cblas.copy(v_cblas, ind)
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(v_cblas.data() - v_numpy.data())/s
    print('error: %e' % t)
    print('time: %.2e' % elapsed)

    print('----\n testing cublas indexed copy...')
    start = time.time()
    u_cublas.copy(v_cublas, ind)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(v_cublas.data() - v_numpy.data())/s
    print('error: %e' % t)
    print('time: %.2e' % elapsed)

    scale = numpy.ones(m)*3.0

    print('----\n testing numpy scale...')
    start = time.time()
    u_numpy.scale(scale)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(u_numpy.data())
    print('time: %.2e' % elapsed)

    print('----\n testing cblas scale...')
    start = time.time()
    u_cblas.scale(scale)
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(u_cblas.data() - u_numpy.data())/s
    print('error: %e' % t)
    print('time: %.2e' % elapsed)

    print('----\n testing cublas scale...')
    start = time.time()
    u_cublas.scale(scale)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(u_cublas.data() - u_numpy.data())/s
    print('error: %e' % t)
    print('time: %.2e' % elapsed)

    print('----\n testing numpy dots...')
    start = time.time()
    p = u_numpy.dots(v_numpy)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(p)
    print('time: %.2e' % elapsed)

    print('----\n testing cblas dots...')
    start = time.time()
    q = u_cblas.dots(v_cblas)
    stop = time.time()
    elapsed = stop - start
    print('error: %e' % (nla.norm(q - p)/s))
    print('time: %.2e' % elapsed)

    print('----\n testing cublas dots...')
    start = time.time()
    q = u_cublas.dots(v_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('error: %e' % (nla.norm(q - p)/s))
    print('time: %.2e' % elapsed)

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
    s = nla.norm(v_numpy.data())

    print('----\n testing cblas multiply...')
    start = time.time()
    u_cblas.multiply(p, v_cblas)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = nla.norm(v_cblas.data() - v_numpy.data())/s
    print('error: %e' % t)

    print('----\n testing cublas multiply...')
    start = time.time()
    u_cublas.multiply(p, v_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = nla.norm(v_cublas.data() - v_numpy.data())/s
    print('error: %e' % t)

    print('----\n testing numpy add...')
    start = time.time()
    v_numpy.add(u_numpy, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = nla.norm(v_numpy.data())/s
    print('error: %e' % t)

    print('----\n testing cblas add...')
    start = time.time()
    v_cblas.add(u_cblas, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = nla.norm(v_cblas.data())/s
    print('error: %e' % t)

    print('----\n testing cublas add...')
    start = time.time()
    v_cublas.add(u_cublas, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    print('time: %.2e' % elapsed)
    t = nla.norm(v_cublas.data())/s
    print('error: %e' % t)

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

except error as err:
    print('%s' % err.value)
