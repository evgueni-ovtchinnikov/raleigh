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

numpy.random.seed(1) # make results reproducible

LOAD = False
SAVE = False
WITH_RESTARTS = False

if LOAD:
    u = numpy.load('C:/Users/wps46139/Documents/Data/PCA/u10K4K.npy')
    v = numpy.load('C:/Users/wps46139/Documents/Data/PCA/v10K4K.npy')
    m, n = u.shape
else:
    # generate the matrix
    m = 1000
    n = 400
    u, v = random_singular_vectors(m, n, numpy.float32)
    if SAVE:
        numpy.save('u.npy', u)
        numpy.save('v.npy', v)
k = min(m, n)
alpha = 0.05
sigma = lambda t: 2**(-alpha*t).astype(numpy.float32)
#alpha = 0.01
#sigma = lambda t: 2**(-alpha*t*t).astype(numpy.float32)
s = random_singular_values(k, sigma, numpy.float32)
a = numpy.dot(u*s, v.transpose())

th = 0.01
block_size = 40

# set solver options
opt = Options()
opt.block_size = block_size
opt.max_iter = 300
opt.verbosity = 1
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-4)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(th, relative = False)

# compute partial svd

# using sliding window scheme
start = time.time()
sigma0, u, vt = partial_svd(a, opt)
stop = time.time()
time_sw = stop - start
iter_sw = opt.stopping_criteria.iteration

# with restarts
#opt.stopping_criteria.set_how_many(block_size)
u = None
vt = None
iter_wr = 0
start = time.time()
while WITH_RESTARTS:
    nsv = int(round(block_size*0.8))
    sigma, u, vt = partial_svd(a, opt, u, vt, nsv)
#    print(u.shape)
#    print(vt.shape)
    iter_wr += opt.stopping_criteria.iteration
#    print(sigma)
    if sigma[-1] < th*sigma[0]:
        break
stop = time.time()
time_wr = stop - start

print(sigma0)
print('iterations: sliding window %d, with restarts %d' % (iter_sw, iter_wr))
print('time: sliding window %.1e, with restarts %.1e' % (time_sw, time_wr))
