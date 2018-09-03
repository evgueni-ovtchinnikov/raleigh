# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:50:23 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla
import sys
import time

raleigh_path = '../../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from random_matrix_for_svd import random_matrix_for_svd
from raleigh.piv_chol import piv_chol

def conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj().T
    else:
        return a.T

def piv_chol_new(A, k, eps, blk = 4, verb = 0):
    n = A.shape[0]
    ind = [i for i in range(n)]
    if k > 0:
        U = sla.cholesky(A[:k,:k])
        A[:k,:k] = U.copy()
        A[:k, k : n] = sla.solve_triangular \
                       (conjugate(U), A[:k, k : n], lower = True)
        A[k : n, :k].fill(0.0)
        A[k : n, k : n] -= numpy.dot(conjugate(A[:k, k : n]), A[:k, k : n])
    l = k
    for i in range(k, n):
        s = numpy.diag(A[i : n, i : n]).copy()
        if i > l:
            t = nla.norm(A[l : i, i : n], axis = 0)
            s -= t*t
        j = i + numpy.argmax(s)
        if i != j:
            #print('swapping %d and %d' % (i, j))
            A[i, :], A[j, :] = A[j, :], A[i, :].copy()
            A[:, i], A[:, j] = A[:, j], A[:, i].copy()
            ind[i], ind[j] = ind[j], ind[i]
        if i > l:
            A[i, i : n] -= numpy.dot(conjugate(A[l : i, i]), A[l : i, i : n])
        last_piv = A[i, i]
        #print('pivot: %e' % last_piv)
        if A[i, i] <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, last_piv
        A[i, i] = math.sqrt(abs(A[i, i]))
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
        lmin = estimate_lmin(A[: i + 1, : i + 1])
        lmax = estimate_lmax(A[: i + 1, : i + 1])
        if verb > 0:
            print('%e %e %e' % (A[i,i], lmin, lmax))
        if lmin/lmax <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, last_piv
        if i - l == blk - 1 and i < n - 1:
            j = i + 1
            A[j : n, j : n] -= numpy.dot(conjugate(A[l : j, j : n]), A[l : j, j : n])
            l += blk
    return ind, 0, A[n - 1, n - 1]**2

def estimate_lmax(U):
    U = numpy.triu(U)
    return sla.norm(numpy.dot(conjugate(U), U), ord = 1)
def estimate_lmin(U):
    n = U.shape[0]
    if U.dtype.kind == 'c':
        tr = 2
    else:
        tr = 1
    x = numpy.ones((n,), dtype = U.dtype)
    s = numpy.dot(x, x)
    for i in range(3):
        y = sla.solve_triangular(U, x, trans = tr)
        t = numpy.dot(y, y)
        rq = s/t
        #print(i, rq)
        x = sla.solve_triangular(U, y)
        s = numpy.dot(x, x)
    return rq

numpy.random.seed(1) # make results reproducible

dtype = numpy.float32
#dtype = numpy.float64

EPS = 0 # 1e-3

m = 10
n = 21
k = 10
l = 10

print('\n--- generating the matrix...')
alpha = 0.05
f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
sigma, u, v0, w = random_matrix_for_svd(m, n, k, f_sigma, dtype)
sigma, u, v, w0 = random_matrix_for_svd(m, n, k, f_sigma, dtype)
print(v.shape)
w = numpy.concatenate((v.T[:l,:], w))
print(w.shape)
s = 1/nla.norm(w, axis = 1)
w = numpy.dot(numpy.diag(s), w)
print(nla.norm(w, axis = 1))

g = numpy.dot(w, w.T)
#print(g[:l,:l])
lmd, q = sla.eigh(g)
print(lmd)

u = g.copy()
#ind, dropped, last_piv = piv_chol(u, l, 1e-2, verb = 1)
ind, dropped, last_piv = piv_chol_new(u, l, 1e-2, blk = 4, verb = 1)
print(ind)
print(dropped)
print(last_piv)
s = nla.norm(g)
h = numpy.dot(conjugate(u), u)
g = g[ind,:]
g = g[:,ind]
g -= h
s = nla.norm(g)/s
print(s)