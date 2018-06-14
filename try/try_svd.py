# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:50:19 2018

@author: Evgueni Ovtchinnikov, STFC
"""
import numpy
import sys
import time
sys.path.append('..')

from raleigh.solver import Options
from raleigh.ndarray.svd import partial_svd
from random_matrix_for_svd import random_singular_values, random_singular_vectors

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

LOAD = False
SAVE = False
WITH_RESTARTS = True
EXP = 1

if LOAD:
    u0 = numpy.load('C:/Users/wps46139/Documents/Data/PCA/u10K4K.npy')
    v0 = numpy.load('C:/Users/wps46139/Documents/Data/PCA/v10K4K.npy')
    m, n = u0.shape
else:
    # generate the matrix
    m = 2000
    n = 4000
    u0, v0 = random_singular_vectors(m, n, min(m, n), numpy.float32)
    if SAVE:
        numpy.save('u.npy', u0)
        numpy.save('v.npy', v0)
k = min(m, n)

if EXP == 1:
    alpha = 0.05
    f_sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
else:
    alpha = 0.01
    f_sigma = lambda t: 2**(-alpha*t*t).astype(numpy.float32)
sigma0 = random_singular_values(k, f_sigma, numpy.float32)
a = numpy.dot(u0*sigma0, v0.transpose())

th = 0.01
block_size = 16

# set solver options
opt = Options()
opt.block_size = block_size
opt.max_iter = 300
#opt.verbosity = 1
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-4)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(th, relative = False)

# compute partial svd

# using sliding window scheme
start = time.time()
sigma, u, vt = partial_svd(a, opt)
stop = time.time()
time_sw = stop - start
iter_sw = opt.stopping_criteria.iteration
n_sw = vt.shape[0]
err_sw = vec_err(v0[:,:n_sw], vt.transpose())

# with restarts
nsv = int(round(block_size*0.8))
opt.stopping_criteria.set_how_many(nsv)
u_wr = None
vt_wr = None
cstr = None
iter_wr = 0
start = time.time()
while WITH_RESTARTS:
#    sigma_wr, u_wr, vt_wr = partial_svd(a, opt, uc = u_wr, vtc = vt_wr)
    sigma_wr, u_wr, vt_wr = partial_svd(a, opt, cstr = cstr)
    iter_wr += opt.stopping_criteria.iteration
#    print(sigma_wr)
#    if iter_wr > 100:
#        break
    if sigma_wr[-1] < th*sigma_wr[0]:
        break
    cstr = (u_wr, vt_wr)
    print('\nrestarting...')
stop = time.time()
time_wr = stop - start

#print(sigma_wr)
#print(sigma)

print('\nsingular values:')
print(sigma[:n_sw + 1])
print('\nsingular vector errors:')
print(err_sw)

print('\niterations: sliding window %d, with restarts %d' % (iter_sw, iter_wr))
print('time: sliding window %.1e, with restarts %.1e' % (time_sw, time_wr))
