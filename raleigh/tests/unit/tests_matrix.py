# -*- coding: utf-8 -*-
'''Unit tests for matrix and vectors algebra.

Usage:
  tests_matrix [-h | --help | <m> <n> <k> <t>]

Arguments:
  m   matrix rows
  n   matrix columns
  k   number of vectors
  t   data type (s/d/c/z)

Created on Tue Aug 14 14:06:19 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

import numpy
import numpy.linalg as nla
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)


def test(u, v):

    try:
        from raleigh.algebra.dense_cblas import Vectors as cblasVectors
        from raleigh.algebra.dense_cblas import Matrix as cblasMatrix
        have_cblas = True
    except:
        have_cblas = False
    try:
        import raleigh.algebra.cuda as cuda
        from raleigh.algebra.dense_cublas import Vectors as cublasVectors
        from raleigh.algebra.dense_cublas import Matrix as cublasMatrix
        have_cublas = True
    except:
        have_cublas = False
    if not have_cblas and not have_cublas:
        return

    n = u.shape[1]
    m = v.shape[1]
    ones = numpy.ones((m, n), dtype=u.dtype)
    
    if have_cblas:
        u_cblas = cblasVectors(u)
        v_cblas = cblasVectors(v)
        x_cblas = cblasVectors(u_cblas)
        y_cblas = cblasVectors(v_cblas)

    if have_cublas:
        mu, nu = u.shape
        mv, nv = v.shape
        u_cublas = cublasVectors(nu, mu, u.dtype)
        v_cublas = cublasVectors(nv, mv, v.dtype)
        a_cublas = cublasMatrix(ones)
        start = time.time()
        u_cublas.fill(u)
        v_cublas.fill(v)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('    cublas vectors uploading time: %.2e' % elapsed)    
        x_cublas = cublasVectors(u_cublas)
        y_cublas = cublasVectors(v_cublas)    

    if have_cblas:
        a_cblas = cblasMatrix(ones)
    if have_cublas:
        cuda.synchronize()
        start = time.time()
        a_cublas.fill(ones)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('    cublas matrix uploading time: %.2e' % elapsed)    

    print('\n--- testing multiplication by a...')    
    if have_cblas:
        start = time.time()
        a_cblas.apply(u_cblas, v_cblas)
        stop = time.time()
        elapsed = stop - start
        print('    cblas time: %.2e' % elapsed)
    if have_cublas:
        start = time.time()
        a_cublas.apply(u_cublas, v_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('    cublas time: %.2e' % elapsed)    
        if have_cblas:
            diff = v_cblas.data() - v_cublas.data()
            print('    error: %.1e' % (nla.norm(diff)/nla.norm(v_cblas.data())))
 
    print('\n--- testing multiplication by a.T...')    
    if have_cblas:
        start = time.time()
        a_cblas.apply(v_cblas, u_cblas, transp=True)
        stop = time.time()
        elapsed = stop - start
        print('    cblas time: %.2e' % elapsed)
    if have_cublas:
        start = time.time()
        a_cublas.apply(v_cublas, u_cublas, transp=True)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('    cublas time: %.2e' % elapsed)        
        diff = u_cblas.data() - u_cublas.data()
        print('    error: %.1e' % (nla.norm(diff)/nla.norm(u_cblas.data())))
    
    if have_cblas:
        b_cblas = cblasOperatorSVD(a_cblas)
    if have_cublas:
        b_cublas = cublasOperatorSVD(a_cublas)
    
    print('\n--- testing multiplication by a.T*a...')    
    if have_cblas:
        start = time.time()
        b_cblas.apply(u_cblas, x_cblas)
        stop = time.time()
        elapsed = stop - start
        print('    cblas time: %.2e' % elapsed)
    if have_cublas:
        start = time.time()
        b_cublas.apply(u_cublas, x_cublas)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('    cublas time: %.2e' % elapsed)    
        if have_cblas:
            diff = x_cblas.data() - x_cublas.data()
            print('    error: %.1e' % (nla.norm(diff)/nla.norm(x_cblas.data())))
    
    print('\n--- testing multiplication by a*a.T...')    
    if have_cblas:
        start = time.time()
        b_cblas.apply(v_cblas, y_cblas, transp=True)
        stop = time.time()
        elapsed = stop - start
        print('    cblas time: %.2e' % elapsed)
    if have_cublas:
        start = time.time()
        b_cublas.apply(v_cublas, y_cublas, transp=True)
        cuda.synchronize()
        stop = time.time()
        elapsed = stop - start
        print('    cublas time: %.2e' % elapsed)    
        if have_cblas:
            diff = y_cblas.data() - y_cublas.data()
            print('    error: %.1e' % (nla.norm(diff)/nla.norm(y_cblas.data())))


class cblasOperatorSVD:
    def __init__(self, op):
        self.op = op
    def apply(self, x, y, transp=False):
        from raleigh.algebra.dense_cblas import Vectors as cblasVectors
        m, n = self.op.shape()
        k = x.nvec()
        if transp:
            z = cblasVectors(n, k, x.data_type())
            self.op.apply(x, z, transp=True)
            self.op.apply(z, y)
        else:
            z = cblasVectors(m, k, x.data_type())
            self.op.apply(x, z)
            self.op.apply(z, y, transp=True)


class cublasOperatorSVD:
    def __init__(self, op):
        self.op = op
    def apply(self, x, y, transp=False):
        from raleigh.algebra.dense_cublas import Vectors as cublasVectors
        m, n = self.op.shape()
        k = x.nvec()
        if transp:
            z = cublasVectors(n, k, x.data_type())
            self.op.apply(x, z, transp=True)
            self.op.apply(z, y)
        else:
            z = cublasVectors(m, k, x.data_type())
            self.op.apply(x, z)
            self.op.apply(z, y, transp=True)


narg = len(sys.argv)
if narg < 2 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\nUsage:\n')
    print('python tests_matrix.py <rows> <columns> <vector_size> <data_type>')
    exit()
n = int(sys.argv[1])
m = int(sys.argv[2])
k = int(sys.argv[3])
dt = sys.argv[4]


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
#    u = numpy.ones((k, n), dtype = dtype)
#    v = numpy.ones((k, m), dtype = dtype)
    u = numpy.random.randn(k, n).astype(dtype)
    v = numpy.random.randn(k, m).astype(dtype)

    if dt == 'c' or dt == 'z':
        print('testing on complex data...')
        test(u + 1j*u, v - 2j*v)
    else:
        print('testing on real data...')
        test(u, v)
        
    print('done')

except Exception as e:
    print(e)
