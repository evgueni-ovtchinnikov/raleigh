# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:50:19 2018

@author: Evgueni Ovtchinnikov, STFC
"""
import numpy
import sys
sys.path.append('..')

from raleigh.solver import Options
from raleigh.ndarray.svd import partial_svd
from random_matrix_for_svd import random_matrix_for_svd

class MyStoppingCriteria:
    def __init__(self):
        self.th = 1.0
    def set_threshold(self, th):
        self.th = th*th
    def satisfied(self, solver):
        if solver.rcon < 1:
            return False
        return numpy.amin(solver.eigenvalues) \
        < self.th*numpy.amax(solver.eigenvalues)

numpy.random.seed(1) # make results reproducible

# generate the matrix
m = 2000
n = 100
alpha = 0.01
sigma = lambda t: 2**(-alpha*t*t).astype(numpy.float32)
s, u, v, a = random_matrix_for_svd(m, n, sigma, numpy.float32)

# set solver options
opt = Options()
opt.block_size = 32
opt.max_iter = 300
opt.verbosity = 2
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-6)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(0.0) # singular value threshold

# compute partial svd
sigma, u, v = partial_svd(a, opt)
print(sigma)
