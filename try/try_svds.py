# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:59:40 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

import numpy
import sys
import time
sys.path.append('..')

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

EXP = 1
EPS = 0 # 1e-3

m = 5000
n = 10000 # 40000
k = 500

if EXP == 1:
    alpha = 0.05
    f_sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
else:
    alpha = 0.01
    f_sigma = lambda t: 2**(-alpha*t*t).astype(numpy.float32)
sigma0, u0, v0, A = random_matrix_for_svd(m, n, k, f_sigma, numpy.float32)
a = 2*numpy.random.rand(m, n).astype(numpy.float32) - 1
s = numpy.linalg.norm(a, axis = 0)
a /= s
A += EPS*a

th = 0.01
block_size = 150 # 128 #144

# set raleigh solver options
opt = Options()
opt.block_size = block_size
#opt.max_iter = 300
opt.verbosity = 1
opt.convergence_criteria.set_error_tolerance \
    ('kinematic eigenvector error', 1e-14)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(th, relative = False)
#opt.stopping_criteria.set_how_many(block_size)

start = time.time()
sigma, u, vt = partial_svd(A, opt)
stop = time.time()
time_r = stop - start
iter_r = opt.stopping_criteria.iteration
n_r = min(sigma.shape[0], sigma0.shape[0])
#n_r = vt.shape[0]
print('\n%d singular values converged in %d iterations' % (sigma.shape[0], iter_r))
if EPS == 0:
#    err_r = vec_err(v0[:,:n_r], vt.transpose())
#    print('\nsingular vector errors (raleigh):')
#    print(err_r)
    print('\nsingular value errors (raleigh):')
    print(abs(sigma[:n_r] - sigma0[:n_r]))
else:
    print('\nsingular values (raleigh):')
    print(sigma)

#sigma = numpy.ndarray((0,), dtype = numpy.float32)
#sigma_max = None
#start = time.time()
#while True:
#    u, s, vt = svds(A, k = block_size)
#    sigma = numpy.concatenate((sigma, s[::-1]))
#    print(s[0])
#    print(s[::-1])
#    if sigma_max is None:
#        sigma_max = numpy.amax(s) # s[-1]
#    if s[0] <= th*sigma_max:
#        break
#    A -= numpy.dot(u*s, vt)
#stop = time.time()
#time_s = stop - start
#
#if EPS == 0:
##    n_s = vt.shape[0]
##    err_s = vec_err(v0[:,:n_s], vt.transpose())
##    print('\nsingular vector errors (svds):')
##    print(err_s[::-1])
#    print('\nsingular value errors (svds):')
#    print(abs(sigma - sigma0[:sigma.shape[0]]))
#else:
#    print('\nsingular values (svds):')
#    print(sigma)
#
#print('\n time: raleigh %.1e, svds %.1e' % (time_r, time_s))
