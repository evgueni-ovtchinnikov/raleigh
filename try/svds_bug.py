# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:44:49 2018

@author: Evgueni Ovtchinnikov, UKRI
"""
import numpy
from scipy.linalg import svd
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

numpy.random.seed(100)
m = 2000
n = 2000
k = 300
alpha = 0.05
print('\n generating the matrix, may take a while...')
f_sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, numpy.float32)
a = 2*numpy.random.rand(m, n).astype(numpy.float32) - 1
s = numpy.linalg.norm(a, axis = 0)
a /= s
A += 5e-3*a
print('\n computing 144 largest singular values...')
u, sigma, vt = svds(A, k = 144)
print('\n 144 largest singular values from svds:')
print(sigma)

u, sigma0, vt = svd(A)
print('\n 144 largest singular values from svd:')
print(sigma0[:144])
print('\n next 5 largest singular values:')
print(sigma0[144:149])
print('\n smallest 5 singular values:')
print(sigma0[-5:])
