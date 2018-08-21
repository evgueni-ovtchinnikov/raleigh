# -*- coding: utf-8 -*-
'''Unit tests for matrix and vectors algebra.

Usage:
  tests_matrix [--help | options] <m> <n> <k>

Arguments:
  m   matrix rows
  n   matrix columns
  k   number of vectors

Options:
  -d, --double
  -c, --complex
'''
'''
Created on Tue Aug 14 14:06:19 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

m = int(args['<m>'])
n = int(args['<n>'])
k = int(args['<k>'])
dble = args['--double']
cmplx = args['--complex']

raleigh_path = '../../..'

import numpy
import numpy.linalg as nla
import sys
import time

if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

import raleigh.cuda.cuda as cuda
#from raleigh.cuda.cublas_algebra import Vectors, Matrix
from raleigh.ndarray.cblas_algebra import Vectors as cblasVectors
from raleigh.ndarray.cblas_algebra import Matrix as cblasMatrix
from raleigh.cuda.cublas_algebra import Vectors as cublasVectors
from raleigh.cuda.cublas_algebra import Matrix as cublasMatrix

class cblasOperatorSVD:
    def __init__(self, op):
        self.op = op
    def apply(self, x, y, transp = False):
        m, n = self.op.shape()
        k = x.nvec()
        if transp:
            z = cblasVectors(n, k, x.data_type())
            self.op.apply(x, z, transp = True)
            self.op.apply(z, y)
        else:
            z = cblasVectors(m, k, x.data_type())
            self.op.apply(x, z)
            self.op.apply(z, y, transp = True)

class cublasOperatorSVD:
    def __init__(self, op):
        self.op = op
    def apply(self, x, y, transp = False):
        m, n = self.op.shape()
        k = x.nvec()
        if transp:
            z = cublasVectors(n, k, x.data_type())
            self.op.apply(x, z, transp = True)
            self.op.apply(z, y)
        else:
            z = cublasVectors(m, k, x.data_type())
            self.op.apply(x, z)
            self.op.apply(z, y, transp = True)

#m = 3
#n = 2

def test(u, v):
    
    u_cblas = cblasVectors(u)
    v_cblas = cblasVectors(v)
    x_cblas = cblasVectors(u_cblas)
    y_cblas = cblasVectors(v_cblas)
    
    u_cublas = cublasVectors(u)
    v_cublas = cublasVectors(v)
    x_cublas = cublasVectors(u_cublas)
    y_cublas = cublasVectors(v_cublas)
    
    n = u_cublas.dimension()
    m = v_cublas.dimension()
    k = u_cublas.nvec()

    a_cblas = cblasMatrix(numpy.ones((m, n), dtype = u_cblas.data_type()))
    a_cublas = cublasMatrix(numpy.ones((m, n), dtype = u_cublas.data_type()))

    print('\n--- testing multiplication by a...')    
    start = time.time()
    a_cblas.apply(u_cblas, v_cblas)
    stop = time.time()
    elapsed = stop - start
    print('    cblas time: %.2e' % elapsed)
    start = time.time()
    a_cublas.apply(u_cublas, v_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('    cublas time: %.2e' % elapsed)    
    diff = v_cblas.data() - v_cublas.data()
    print('    error: %.1e' % (nla.norm(diff)/nla.norm(v_cblas.data())))
#    print(v_cublas.data())
 
    print('\n--- testing multiplication by a.T...')    
    start = time.time()
    a_cblas.apply(v_cblas, u_cblas, transp = True)
    stop = time.time()
    elapsed = stop - start
    print('    cblas time: %.2e' % elapsed)
    start = time.time()
    a_cublas.apply(v_cublas, u_cublas, transp = True)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('    cublas time: %.2e' % elapsed)        
    diff = u_cblas.data() - u_cublas.data()
    print('    error: %.1e' % (nla.norm(diff)/nla.norm(u_cblas.data())))
#    print(u_cublas.data())
    
    b_cblas = cblasOperatorSVD(a_cblas)
    b_cublas = cublasOperatorSVD(a_cublas)
    
    print('\n--- testing multiplication by a.T*a...')    
    start = time.time()
    b_cblas.apply(u_cblas, x_cblas)
    stop = time.time()
    elapsed = stop - start
    print('    cblas time: %.2e' % elapsed)
    start = time.time()
    b_cublas.apply(u_cublas, x_cublas)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('    cublas time: %.2e' % elapsed)    
    diff = x_cblas.data() - x_cublas.data()
    print('    error: %.1e' % (nla.norm(diff)/nla.norm(x_cblas.data())))
#    print(x_cublas.data())
    
    print('\n--- testing multiplication by a*a.T...')    
    start = time.time()
    b_cblas.apply(v_cblas, y_cblas, transp = True)
    stop = time.time()
    elapsed = stop - start
    print('    cblas time: %.2e' % elapsed)
    start = time.time()
    b_cublas.apply(v_cublas, y_cublas, transp = True)
    cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    print('    cublas time: %.2e' % elapsed)    
    diff = y_cblas.data() - y_cublas.data()
    print('    error: %.1e' % (nla.norm(diff)/nla.norm(y_cblas.data())))
#    print(y_cublas.data())

try:
    if dble:
        print('running in double precision...')
        dt = numpy.float64
    else:
        print('running in single precision...')
        dt = numpy.float32
#    u = numpy.ones((k, n), dtype = dt)
#    v = numpy.ones((k, m), dtype = dt)
    u = numpy.random.randn(k, n).astype(dt)
    v = numpy.random.randn(k, m).astype(dt)

    if cmplx:
        print('testing on complex data...')
        test(u + 1j*u, v - 2j*v)
    else:
        print('testing on real data...')
        test(u, v)
        
    print('done')

except Exception as e:
    print(e)
