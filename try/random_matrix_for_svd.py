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
    return sigma(k*(s - t))

def random_singular_vectors(m, n, dt):
    k = min(m, n)
    u = numpy.random.randn(m, k).astype(dt)
    v = numpy.random.randn(n, k).astype(dt)
    u, r = numpy.linalg.qr(u)
    v, r = numpy.linalg.qr(v)
    return u, v

#def random_matrix_for_svd(m, n, u, v, sigma, dt):
#    k = min(m, n)
#    u = numpy.random.randn(m, k).astype(dt)
#    v = numpy.random.randn(n, k).astype(dt)
#    u, r = numpy.linalg.qr(u)
#    v, r = numpy.linalg.qr(v)
#    s = random_singular_values(k, sigma, dt)
#    x = numpy.arange(k)
#    plt.figure()
#    plt.plot(x, s)
#    plt.show()
#    a = numpy.dot(u*s, v.transpose())
#    return s, u, v, a
