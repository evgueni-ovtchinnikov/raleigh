# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:41:23 2018

@author: Evgueni Ovtchinnikov, STFC
"""

import numpy
import matplotlib.pyplot as plt

def random_singular_values(k, sigma, dt):
    s = numpy.random.rand(k).astype(dt)
    s = numpy.sort(s)
    t = numpy.ones(k)*s[0]
    return sigma(s - t)
#    return sigma(k*(s - t))

def random_singular_vectors(m, n, k, dt):
    u = numpy.random.randn(m, k).astype(dt)
    v = numpy.random.randn(n, k).astype(dt)
    u, r = numpy.linalg.qr(u)
    v, r = numpy.linalg.qr(v)
    return u, v

def random_matrix_for_svd(m, n, k, sigma, dt):
#    s = random_singular_values(k, sigma, dt)
    s = random_singular_values(min(m,n), sigma, dt)[:k]
    u, v = random_singular_vectors(m, n, k, dt)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a

