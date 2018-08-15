# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:59:40 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

import numpy
import sys
import time

raleigh_path = '..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)


from scipy.sparse.linalg import svds
from random_matrix_for_svd import random_matrix_for_svd

from raleigh.solver import Options
from raleigh.ndarray.svd import partial_svd

class MyStoppingCriteria:
    def __init__(self):
        self.rel = True
        self.th = 1.0
        self.m = -1
        self.iteration = -1
    def set_threshold(self, th, relative = True):
        self.rel = relative
        self.th = th*th
    def set_how_many(self, m):
        self.m = m
    def satisfied(self, solver):
        self.iteration = solver.iteration
        if solver.rcon < 1:
            return False
        vmin = numpy.amin(solver.eigenvalues)
        if self.rel:
            th = self.th*numpy.amax(solver.eigenvalues)
        else:
            th = self.th
        done = vmin < th or self.m > 0 and solver.rcon >= self.m
        return done

def vec_err(u, v):
    w = v.copy()
    q = numpy.dot(u.T, v)
    w = numpy.dot(u, q) - v
    s = numpy.linalg.norm(w, axis = 0)
    return s

numpy.random.seed(1) # make results reproducible

dtype = numpy.float32
#dtype = numpy.float64

EXP = 1
EPS = 0 # 1e-3

m = 25000
n = 15000 # 40000
k = 500

print('\n--- generating the matrix...')
if EXP == 1:
    alpha = 0.05
    f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
else:
    alpha = 0.01
    f_sigma = lambda t: 2**(-alpha*t*t).astype(dtype)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, dtype)
#a = 2*numpy.random.rand(m, n).astype(dtype) - 1
#s = numpy.linalg.norm(a, axis = 0)
#a /= s
#A += EPS*a

th = 0.01
block_size = 64 #32 #150 # 128 #144
vec_tol = 1e-4 #-26

print('\n--- solving with raleigh.ndarray.svd...')

# set raleigh solver options
opt = Options()
opt.block_size = block_size
opt.max_iter = 400
#opt.verbosity = 2
opt.convergence_criteria.set_error_tolerance \
    ('kinematic eigenvector error', vec_tol)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(th, relative = False)

start = time.time()

sigma, u, vt = partial_svd(A, opt, arch = 'gpu')

stop = time.time()
time_r = stop - start
iter_r = opt.stopping_criteria.iteration
sigma = sigma[sigma > th*sigma[0]]
n_r = min(sigma.shape[0], sigma0.shape[0])
#n_r = vt.shape[0]
print('\n%d singular values converged in %d iterations' % (sigma.shape[0], iter_r))
if EPS == 0:
    err_r = vec_err(v0[:,:n_r], vt.transpose()[:,:n_r])
    print('\nsingular vector errors (raleigh):')
    print(err_r)
    print('\nsingular value errors (raleigh):')
    print(abs(sigma[:n_r] - sigma0[:n_r]))
else:
    print('\nsingular values (raleigh):')
    print(sigma)

print('\n--- solving with restarted scipy.sparse.linalg.svds...')

sigma = numpy.ndarray((0,), dtype = dtype)
vt = numpy.ndarray((0, n), dtype = dtype)
sigma_max = None

start = time.time()

while True:
    u, s, vti = svds(A, k = block_size, tol = vec_tol)
    sigma = numpy.concatenate((sigma, s[::-1]))
    vt = numpy.concatenate((vt, vti[::-1, :]))
    print('last singular value computed: %e' % s[0])
#    print(s[::-1])
    if sigma_max is None:
        sigma_max = numpy.amax(s) # s[-1]
    if s[0] <= th*sigma_max:
        break
    print('deflating...')
    A -= numpy.dot(u*s, vti)
    print('restarting...')

stop = time.time()
time_s = stop - start

sigma = sigma[sigma > th*sigma[0]]
n_s = min(sigma.shape[0], sigma0.shape[0])
if EPS == 0:
    #n_s = vt.shape[0]
    err_s = vec_err(v0[:,:n_s], vt.transpose()[:,:n_s])
    print('\nsingular vector errors (svds):')
    print(err_s) #[::-1])
#    print(err_s.shape, n_s)
    print('\nsingular value errors (svds):')
    print(abs(sigma[:n_s] - sigma0[:n_s]))
#    print(abs(sigma - sigma0[:sigma.shape[0]]))
#else:
print('\nsingular values (svds):')
print(sigma[:n_s])
#print(sigma0[:n_s])

print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))
