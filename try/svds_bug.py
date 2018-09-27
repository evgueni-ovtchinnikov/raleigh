# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:44:49 2018

@author: Evgueni Ovtchinnikov, UKRI
"""
import numpy
from scipy.sparse.linalg import svds

def random_singular_values(k, sigma, dt):
    s = numpy.random.rand(k).astype(dt)
    s = numpy.sort(s)
    t = numpy.ones(k)*s[0]
    return sigma(k*(s - t))

def random_singular_vectors(m, n, k, dt):
    u = numpy.random.randn(m, k).astype(dt)
    v = numpy.random.randn(n, k).astype(dt)
    u, r = numpy.linalg.qr(u)
    v, r = numpy.linalg.qr(v)
    return u, v

def random_matrix_for_svd(m, n, k, sigma, dt):
    u, v = random_singular_vectors(m, n, k, dt)
    s = random_singular_values(k, sigma, dt).astype(dt)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a

numpy.random.seed(1)

m = 5000
n = 10000
l = 400
alpha = 0.05
f_sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, l, f_sigma, numpy.float32)
print('\n 128 largest singular values of A:')
print(sigma0[::-1])
u, sigma, vt = svds(A, k = 128)
print('\n 128 largest singular values from svds:')
print(sigma)

# another test: this time A is unlikely to have multiple zero singular values
m = 5000
n = 40000
k = 500
alpha = 0.05
print('\n generating the matrix, may take a while...')
f_sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, numpy.float32)
a = 2*numpy.random.rand(m, n).astype(numpy.float32) - 1
s = numpy.linalg.norm(a, axis = 0)
a /= s
A += 1e-3*a
print('\n computing 144 largest singular values...')
u, sigma, vt = svds(A, k = 144)
print('\n 144 largest singular values from svds:')
print(sigma)

# another test: svds computes rubbish
numpy.random.seed(1)
m = 16000
n = 16000
k = 20
alpha = 0.01
print('\n generating the matrix, may take a while...')
f_sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, numpy.float32)
a = 2*numpy.random.rand(m, n).astype(numpy.float32) - 1
s = numpy.linalg.norm(a, axis = 0)
A += a*(min(sigma0[-1], 1e-3)/s)
print('\n computing 128 largest singular values...')
u, sigma, vt = svds(A, k = 128)
print('\n 128 largest singular values from svds:')
print(sigma)