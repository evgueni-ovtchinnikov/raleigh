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
matrix_path = '../matrix'
if matrix_path not in sys.path:
    sys.path.append(matrix_path)

from random_matrix_for_svd import random_matrix_for_svd
from raleigh.piv_chol import piv_chol

def conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj().T
    else:
        return a.T

def piv_chol_new(A, k, eps, blk = 64, verb = 0):
    n = A.shape[0]
    buff = A[0,:].copy()
    ind = [i for i in range(n)]
    drop_case = 0
    dropped = 0
    last_check = -1
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
            buff[:] = A[i,:]
            A[i,:] = A[j,:]
            A[j,:] = buff
            buff[:] = A[:,i]
            A[:,i] = A[:,j]
            A[:,j] = buff
            ind[i], ind[j] = ind[j], ind[i]
        if i > l:
            A[i, i : n] -= numpy.dot(conjugate(A[l : i, i]), A[l : i, i : n])
        last_piv = A[i, i].real
        if last_piv <= eps:
            A[i : n, :].fill(0.0)
            drop_case = 1
            dropped = n - i
            break
        A[i, i] = math.sqrt(last_piv)
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
        if i - l == blk - 1 or i == n - 1:
            last_check = i
            lmin = estimate_lmin(A[: i + 1, : i + 1])
            lmax = estimate_lmax(A[: i + 1, : i + 1])
            if verb > 0:
                print('%e %e %e' % (A[i,i], lmin, lmax))
            if lmin/lmax <= eps:
                A[i : n, :].fill(0.0)
                drop_case = 2
                dropped = n - i
                break
        if i - l == blk - 1 and i < n - 1:
            j = i + 1
            A[j : n, j : n] -= numpy.dot(conjugate(A[l : j, j : n]), A[l : j, j : n])
            l += blk
    if last_check < n - 1 and drop_case != 2:
        i = last_check
        j = n - dropped - 1
        while i < j:
            m = i + (j - i + 1)//2
            lmin = estimate_lmin(A[: m + 1, : m + 1])
            lmax = estimate_lmax(A[: m + 1, : m + 1])
            if verb > 0:
                print('%d %e %e' % (m, lmin, lmax))
            if lmin/lmax <= eps:
                if j > m:
                    j = m
                    continue
                else:
                    A[j : n, :].fill(0.0)
                    dropped = n - j
                    last_piv = A[j - 1, j - 1]**2
                    break
            else:
                i = m
                continue
    return ind, dropped, last_piv

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

m = 1000
n = 4000
k = 1000
l = 1000

print('\n--- generating the matrix...')
alpha = 0.05
f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
sigma, u, v0, w = random_matrix_for_svd(m, n, k, f_sigma, dtype)
sigma, u, v, w0 = random_matrix_for_svd(m, n, k, f_sigma, dtype)
print(v.shape)
print(w.shape)
w = numpy.concatenate((v.T[:l,:], w))
print(w.shape)
s = 1/nla.norm(w, axis = 1)
w = numpy.dot(numpy.diag(s), w)
#print(nla.norm(w, axis = 1))

g = numpy.dot(w, w.T)
#print(g[:l,:l])
lmd, q = sla.eigh(g)
print(lmd[0], lmd[-1])

u = g.copy()
print('factorizing...')
start = time.time()
ind, dropped, last_piv = piv_chol(u, l, 1e-4, verb = 0)
#ind, dropped, last_piv = piv_chol_new(u, l, 1e-4, blk = 64, verb = 0)
stop = time.time()
print('time: %.1e' % (stop - start))
#print(ind)
print(dropped)
print(last_piv)
s = nla.norm(g)
k = u.shape[0] - dropped
h = numpy.dot(conjugate(u[:k, :k]), u[:k, :k])
g = g[ind,:]
g = g[:,ind]
g[:k, :k] -= h
s = nla.norm(g[:k, :k])/s
print(s)
