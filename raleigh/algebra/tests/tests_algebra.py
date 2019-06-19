'''Unit tests for vectors algebra.

Usage:
  python tests_algebra [-h | --help | <dim> <nv> <dtype>]

Arguments:
  dim    vector size
  nv     number of vectors
  dtype  data type (s/d/c/z)
'''

import numpy
import numpy.linalg as nla
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)


def test(u, v):

    from raleigh.algebra.dense_numpy import Vectors as numpyVectors
    try:
        from raleigh.algebra.dense_cblas import Vectors as cblasVectors
        have_cblas = True
    except:
        have_cblas = False
    try:
        import raleigh.algebra.cuda_wrap as cuda
        from raleigh.algebra.dense_cublas import Vectors as cublasVectors
        have_cublas = True
    except:
        have_cublas = False

    u_numpy = numpyVectors(u)
    v_numpy = numpyVectors(v)

    if have_cblas:
        u_cblas = cblasVectors(u.copy())
        v_cblas = cblasVectors(v.copy())

    if have_cublas:
        u_cublas = cublasVectors(u)
        v_cublas = cublasVectors(v)
        w_cublas = cublasVectors(v)
        print('----\n testing cublasVectors.zero...')
        w_cublas.zero()
        t = nla.norm(w_cublas.data())
        print('error: %e' % t)
        print('----\n testing cublasVectors.fill_random...')
        w_cublas.fill_random()
        w_data = w_cublas.data()
        print(numpy.mean(w_data))
        print(numpy.var(w_data))

    print('----\n testing numpy copy...')
    start = time.time()
    u_numpy.copy(v_numpy)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(v_numpy.data())
    print('time: %.2e' % elapsed)

    if have_cblas:
        print('----\n testing cblas copy...')
        start = time.time()
        u_cblas.copy(v_cblas)
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cblas.data() - v_numpy.data())/s
        print('error: %e' % t)
        print('time: %.2e' % elapsed)

    if have_cublas:
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

    if have_cblas:
        print('----\n testing cblas indexed copy...')
        start = time.time()
        u_cblas.copy(v_cblas, ind)
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cblas.data() - v_numpy.data())/s
        print('error: %e' % t)
        print('time: %.2e' % elapsed)

    if have_cublas:
        print('----\n testing cublas indexed copy...')
        start = time.time()
        u_cublas.copy(v_cublas, ind)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cublas.data() - v_numpy.data())/s
        print('error: %e' % t)
        print('time: %.2e' % elapsed)

    scale = numpy.ones(m)*2.0
    multiply = True

    print('----\n testing numpy scale...')
    start = time.time()
    u_numpy.scale(scale, multiply)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(u_numpy.data())
    print('time: %.2e' % elapsed)

    if have_cblas:
        print('----\n testing cblas scale...')
        start = time.time()
        u_cblas.scale(scale, multiply)
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(u_cblas.data() - u_numpy.data())/s
        print('error: %e' % t)
        print('time: %.2e' % elapsed)

    if have_cublas:
        print('----\n testing cublas scale...')
        start = time.time()
        u_cublas.scale(scale, multiply)
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

    if have_cblas:
        print('----\n testing cblas dots...')
        start = time.time()
        q = u_cblas.dots(v_cblas)
        stop = time.time()
        elapsed = stop - start
        print('error: %e' % (nla.norm(q - p)/s))
        print('time: %.2e' % elapsed)

    if have_cublas:
        print('----\n testing cublas dots...')
        start = time.time()
        q = u_cublas.dots(v_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('error: %e' % (nla.norm(q - p)/s))
        print('time: %.2e' % elapsed)

    print('----\n testing numpy transposed dots...')
    start = time.time()
    p = u_numpy.dots(v_numpy, transp=True)
    stop = time.time()
    elapsed = stop - start
    s = nla.norm(p)
    print('time: %.2e' % elapsed)

    if have_cblas:
        print('----\n testing cblas transposed dots...')
        start = time.time()
        q = u_cblas.dots(v_cblas, transp=True)
        stop = time.time()
        elapsed = stop - start
        print('error: %e' % (nla.norm(q - p)/s))
        print('time: %.2e' % elapsed)

    if have_cublas:
        print('----\n testing cublas transposed dots...')
        start = time.time()
        q = u_cublas.dots(v_cublas, transp=True)
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

    if have_cblas:
        print('----\n testing cblas dot...')
        start = time.time()
        q = u_cblas.dot(v_cblas)
        stop = time.time()
        elapsed = stop - start
        print('error: %e' % (nla.norm(q - p)/s))
        print('time: %.2e' % elapsed)

    if have_cublas:
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

    if have_cblas:
        print('----\n testing cblas multiply...')
        start = time.time()
        u_cblas.multiply(p, v_cblas)
        stop = time.time()
        elapsed = stop - start
        print('time: %.2e' % elapsed)
        t = nla.norm(v_cblas.data() - v_numpy.data())/s
        print('error: %e' % t)

    if have_cublas:
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

    if have_cblas:
        print('----\n testing cblas add...')
        start = time.time()
        v_cblas.add(u_cblas, -1.0, p)
        stop = time.time()
        elapsed = stop - start
        print('time: %.2e' % elapsed)
        t = nla.norm(v_cblas.data())/s
        print('error: %e' % t)

    if have_cublas:
        print('----\n testing cublas add...')
        start = time.time()
        v_cublas.add(u_cublas, -1.0, p)
        stop = time.time()
        elapsed = stop - start
        print('time: %.2e' % elapsed)
        t = nla.norm(v_cublas.data())/s
        print('error: %e' % t)

    z_numpy = cblasVectors(u_numpy, shallow=True)
    z_numpy.zero()
    q = u_numpy.dots(u_numpy)
    print(nla.norm(q))
    del u_numpy
    q = z_numpy.dots(z_numpy)
    print(nla.norm(q))

    if have_cblas:
        z_cblas = cblasVectors(u_cblas, shallow=True)
        z_cblas.zero()
        q = u_cblas.dots(u_cblas)
        print(nla.norm(q))
        del u_cblas
        q = z_cblas.dots(z_cblas)
        print(nla.norm(q))

    if have_cublas:
        z_cublas = cublasVectors(u_cublas, shallow=True)
        z_cublas.zero()
        q = u_cublas.dots(u_cublas)
        print(nla.norm(q))
        del u_cublas
        q = z_cublas.dots(z_cublas)
        print(nla.norm(q))


narg = len(sys.argv)
if narg < 2 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\nUsage:\n')
    print('python tests_algebra.py <vector_size> <number_of_vectors> <data_type>')
    exit()
n = int(sys.argv[1])
m = int(sys.argv[2])
dt = sys.argv[3]

numpy.random.seed(1) # make results reproducible

try:
    if dt == 's':
        dtype = numpy.float32
    elif dt == 'd':
        dtype = numpy.float64
    elif dt == 'c':
        dtype = numpy.complex64
    elif dt == 'z':
        dtype = numpy.complex128
    else:
        raise ValueError('data type %s not supported' % dt)

#    u = numpy.ones((m, n), dtype = dtype)
#    v = numpy.ones((m, n), dtype = dtype)
    u = numpy.random.randn(m, n).astype(dtype)
    v = numpy.random.randn(m, n).astype(dtype)

    if dt == 'c' or dt == 'z':
        print('testing on complex data...')
        test(u + 1j*v, v - 2j*u)
    else:
        print('testing on real data...')
        test(u, v)
        
    print('done')

except Exception as e:
    print(e)
