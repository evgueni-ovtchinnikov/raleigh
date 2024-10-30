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
import os
import scipy.linalg as sla
import sys
import time

from raleigh.algebra import verbosity
verbosity.level = 2

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


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a


def test_lra_ortho(u, v, wu, wv):

    # have: vector sets u = [u_1, ..., u_k] and v = [v_1, .., v_k]
    # want: orthonormal vector set u' and orthogonal vector set v' such that
    # 1) span(u) = span(u'), span(v) = span(v')
    # 2) v*u.H = v'*u'.H

    print('transfom via svd for R...')
    u.copy(wu)
    s, q = wu.svd() # u == wu*s*q, wu: orthonormal set, s: diag, q: unitary
    v.multiply(q, wv)
    wv.scale(s, multiply=True) # vw = v*q.H*s
    # theory: wv*wu.H = v*q.H*s*wu.H = v*(wu*s*q.H) = v*u.H
    # let us check numerically by measuring D = wv*wu.H - v*u.H
    # D*w = 0 for any w orthogonal to span(u) = span(wu)
    # hence enough to measure D*wu = wv - v*u.H*wu
    p = wu.dot(u) # p = u.H*wu
    t = wv.dots(wv)
    wv.add(v, -1.0, p) # wv := wv - v*u.H*wu 
    t = numpy.sqrt(wv.dots(wv)/t)
    print('transformation error: %.1e' % nla.norm(t))
    # now make wv orthogonal
    wv.add(v, 1.0, p) # restore wv 
    print('transfom via svd for L...')
    wv.copy(v) # use v as a workspace
    s, q = v.svd() # wv == v*s*q
    wu.multiply(q, u) # u' = wu*q.H: orthonormal because q is unitary
    v.scale(s, multiply=True) # v' = v*s
    # theory: v'*u'.H = v*s*(wu*q.H).H = v*s*q*wu.H = wv*wu.H
    # let us check numerically by measuring D = wv*wu.H - v'*u'.H
    # D*w = 0 for any w orthogonal to span(u) = span(wu) = span(u')
    # hence enough to measure D*wu = wv - v'*u'.H*wu
    p = wu.dot(u) # p = u'.H*wu
    t = wv.dots(wv)
    wv.add(v, -1.0, p) # wv := wv - v'*u'.H*wu
    t = numpy.sqrt(wv.dots(wv)/t)
    print('transformation error: %.1e' % nla.norm(t))
    q = u.dot(u)
    lmd, x = sla.eigh(q)
    print('R non-orthonormality: %.1e' % (lmd[-1]/lmd[0] - 1.0))


def test1(u, v):

    u_numpy = numpyVectors(u.copy())
    v_numpy = numpyVectors(v.copy())
    w_numpy = numpyVectors(v.copy())
    x_numpy = numpyVectors(v.copy())

    if have_cblas:
        u_cblas = cblasVectors(u.copy())
        v_cblas = cblasVectors(v.copy())
        w_cblas = cblasVectors(v.copy())
        x_cblas = cblasVectors(v.copy())

    if have_cublas:
        u_cublas = cublasVectors(u)
        v_cublas = cublasVectors(v)
        w_cublas = cublasVectors(v)
        x_cublas = cublasVectors(v)

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
#        t = nla.norm(v_cblas.data() - v_numpy.data())/s
        t = nla.norm(v_cblas.data() - u_cblas.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas copy...')
        start = time.time()
        u_cublas.copy(v_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
#        t = nla.norm(v_cublas.data() - v_numpy.data())/s
        t = nla.norm(v_cublas.data() - u_cublas.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

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
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas indexed copy...')
        start = time.time()
        u_cublas.copy(v_cublas, ind)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cublas.data() - v_numpy.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

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
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas scale...')
        start = time.time()
        u_cublas.scale(scale, multiply)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(u_cublas.data() - u_numpy.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

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
        t = nla.norm(q - p)/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas dots...')
        start = time.time()
        q = u_cublas.dots(v_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(q - p)/s
        print('error: %e, time: %.2e' % (t, elapsed))

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
        t = nla.norm(q - p)/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas transposed dots...')
        start = time.time()
        q = u_cublas.dots(v_cublas, transp=True)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(q - p)/s
        print('error: %e, time: %.2e' % (t, elapsed))

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
        t = nla.norm(q - p)/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas dot...')
        start = time.time()
        q = u_cublas.dot(v_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(q - p)/s
        print('error: %e, time: %.2e' % (t, elapsed))

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
        t = nla.norm(v_cblas.data() - v_numpy.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas multiply...')
        start = time.time()
        u_cublas.multiply(p, v_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cublas.data() - v_numpy.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    print('----\n testing numpy add...')
    start = time.time()
    v_numpy.add(u_numpy, -1.0, p)
    stop = time.time()
    elapsed = stop - start
    t = nla.norm(v_numpy.data())/s
    print('error: %e, time: %.2e' % (t, elapsed))

    if have_cblas:
        print('----\n testing cblas add...')
        start = time.time()
        v_cblas.add(u_cblas, -1.0, p)
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cblas.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas add...')
        start = time.time()
        v_cublas.add(u_cublas, -1.0, p)
        stop = time.time()
        elapsed = stop - start
        t = nla.norm(v_cublas.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    print('----\n testing numpy vector reference...')
    nv = u_numpy.nvec()//2
    z_numpy = u_numpy.reference()
    z_numpy.select(nv, nv)
    q = u_numpy.dots(u_numpy)
    print(nla.norm(q))
    z_numpy.zero()
    q = u_numpy.dots(u_numpy)
    print(nla.norm(q))
    # save and delete u_numpy
    u_numpy.copy(w_numpy)
    del u_numpy
    # reference still there
    q = z_numpy.dots(z_numpy)
    print(nla.norm(q))
    # restore u_numpy
    z_numpy.select_all()
    u_numpy = z_numpy

    if have_cblas:
        print('----\n testing cblas vector reference...')
        z_cblas = u_cblas.reference()
        z_cblas.select(nv, nv)
        q = u_cblas.dots(u_cblas)
        print(nla.norm(q))
        z_cblas.zero()
        q = u_cblas.dots(u_cblas)
        print(nla.norm(q))
        u_cblas.copy(w_cblas)
        del u_cblas
        q = z_cblas.dots(z_cblas)
        print(nla.norm(q))
        z_cblas.select_all()
        u_cblas = z_cblas

    if have_cublas:
        print('----\n testing cublas vector reference...')
        z_cublas = u_cublas.reference()
        z_cublas.select(nv, nv)
        q = u_cublas.dots(u_cublas)
        print(nla.norm(q))
        z_cublas.zero()
        q = u_cublas.dots(u_cublas)
        print(nla.norm(q))
        u_cublas.copy(w_cublas)
        del u_cublas
        q = z_cublas.dots(z_cublas)
        print(nla.norm(q))
        z_cublas.select_all()
        u_cublas = z_cublas

    print('----\n testing numpy svd...')
    w_numpy.copy(u_numpy)
    s = nla.norm(u_numpy.data())
    start = time.time()
    sigma, q = w_numpy.svd()
    stop = time.time()
    elapsed = stop - start
    w_numpy.scale(sigma, multiply=True)
    w_numpy.multiply(_conj(q).T, v_numpy)
    u_numpy.add(v_numpy, -1.0)
    t = nla.norm(u_numpy.data())/s
    print('error: %e, time: %.2e' % (t, elapsed))

    if have_cblas:
        print('----\n testing cblas svd...')
        w_cblas.copy(u_cblas)
        s = nla.norm(u_cblas.data())
        start = time.time()
        sigma, q = w_cblas.svd()
        stop = time.time()
        elapsed = stop - start
        w_cblas.scale(sigma, multiply=True)
        w_cblas.multiply(_conj(q).T, v_cblas)
        u_cblas.add(v_cblas, -1.0)
        t = nla.norm(u_cblas.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    if have_cublas:
        print('----\n testing cublas svd...')
        w_cublas.copy(u_cublas)
        s = nla.norm(u_cublas.data())
        start = time.time()
        sigma, q = w_cublas.svd()
        stop = time.time()
        elapsed = stop - start
#        print(sigma)
#        print(q.shape, q.dtype)
#        print(sigma.shape, sigma.dtype)
        w_cublas.scale(sigma, multiply=True)
        w_cublas.multiply(_conj(q).T, v_cublas)
        u_cublas.add(v_cublas, -1.0)
        t = nla.norm(u_cublas.data())/s
        print('error: %e, time: %.2e' % (t, elapsed))

    print('----\n testing numpy orthogonalize...')
    w_numpy.fill_orthogonal()
    s = w_numpy.dots(w_numpy)
    s = numpy.sqrt(s)
    w_numpy.scale(s)
    q0 = x_numpy.dot(x_numpy)
    q_numpy = x_numpy.orthogonalize(w_numpy)
    q = x_numpy.dot(w_numpy)
    print('error: %e' % (nla.norm(q)/nla.norm(q0)))

    if have_cblas:
        print('----\n testing cblas orthogonalize...')
        #w_cblas = cblasVectors(w_numpy.data())
        w_cblas.fill(w_numpy.data())
        q0 = x_cblas.dot(x_cblas)
        q_cblas = x_cblas.orthogonalize(w_cblas)
        q = w_cblas.dot(x_cblas)
        print('error: %e' % (nla.norm(q)/nla.norm(q0)))

    if have_cublas:
        #w_cublas = cublasVectors(w_numpy.data())
        w_cublas.fill(w_numpy.data())
        print('----\n testing cublas orthogonalize...')
        q0 = x_cublas.dot(x_cublas)
        q_cublas = x_cublas.orthogonalize(w_cublas)
        q = w_cublas.dot(x_cublas)
        print('error: %e' % (nla.norm(q)/nla.norm(q0)))

    print('----\n testing numpy append axis=1...')
    w_numpy.copy(x_numpy)
    s = x_numpy.dots(x_numpy)
    print(nla.norm(s))
    x_numpy.append(w_numpy, axis=1)
    s = x_numpy.dots(x_numpy)
    print(nla.norm(s))
    x_numpy.append(w_numpy, axis=1)
    s = x_numpy.dots(x_numpy)
    print(nla.norm(s))

    if have_cblas:
        print('----\n testing cblas append axis=1...')
        w_cblas.copy(x_cblas)
        s = x_cblas.dots(x_cblas)
        print(nla.norm(s))
        x_cblas.append(w_cblas, axis=1)
        s = x_cblas.dots(x_cblas)
        print(nla.norm(s))
        x_cblas.append(w_cblas, axis=1)
        s = x_cblas.dots(x_cblas)
        print(nla.norm(s))
        x_cblas.append(w_cblas, axis=1)

    if have_cublas:
        print('----\n testing cublas append axis=1...')
        w_cublas.copy(x_cublas)
        s = w_cublas.dots(w_cublas)
        print(nla.norm(s))
        w_cublas.append(x_cublas, axis=1)
        s = w_cublas.dots(w_cublas)
        print(nla.norm(s))
        w_cublas.append(x_cublas, axis=1)
        s = w_cublas.dots(w_cublas)
        print(nla.norm(s))

    if have_cublas:
        print('----\n testing cublasVectors.zero...')
        w_cublas.zero()
        t = nla.norm(w_cublas.data())
        print('error: %e' % t)
        print('----\n testing cublasVectors.fill_random...')
        w_cublas.fill_random()
        w_data = w_cublas.data()
        print(numpy.mean(w_data))
        print(numpy.var(w_data))


def test2(u, v):

    print('----\n testing orthogonalization of L and R in L R*...')

    u_numpy = numpyVectors(u.copy())
    v_numpy = numpyVectors(v.copy())
    w_numpy = numpyVectors(v.copy())
    x_numpy = numpyVectors(v.copy())
    print('----\n numpy...')
    test_lra_ortho(u_numpy, v_numpy, w_numpy, x_numpy)

    if have_cblas:
        u_cblas = cblasVectors(u.copy())
        v_cblas = cblasVectors(v.copy())
        w_cblas = cblasVectors(v.copy())
        x_cblas = cblasVectors(v.copy())
        print('----\n cblas...')
        test_lra_ortho(u_cblas, v_cblas, w_cblas, x_cblas)

    if have_cublas:
        u_cublas = cublasVectors(u)
        v_cublas = cublasVectors(v)
        w_cublas = cublasVectors(v)
        x_cublas = cublasVectors(v)
        print('----\n cublas...')
        test_lra_ortho(u_cublas, v_cublas, w_cublas, x_cublas)


narg = len(sys.argv)
if narg < 4 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
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
        test1(u + 1j*v, v - 2j*u)
        test2(u + 1j*v, v - 2j*u)
    else:
        print('testing on real data...')
        test1(u, v)
        test2(u, v)
        
    print('done')

except Exception as e:
    print(e)
