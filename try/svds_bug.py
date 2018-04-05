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
print('\n exact singular values:')
print(sigma0[::-1])
u, sigma, vt = svds(A, k = 128)
print('\n singular values from svds:')
print(sigma)