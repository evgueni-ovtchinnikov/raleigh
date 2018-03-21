# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:50:19 2018

@author: Evgueni Ovtchinnikov, UKRI
"""
import numpy
import sys
sys.path.append('..')

from raleigh.solver import Options
from raleigh.ndarray_svd import ndarray_svd

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

def random_svd(m, n, alpha):
    k = min(m, n)
    u = numpy.random.randn(m, k).astype(numpy.float32)
    v = numpy.random.randn(n, k).astype(numpy.float32)
    s = numpy.random.rand(k).astype(numpy.float32)
    u, r = numpy.linalg.qr(u)
    v, r = numpy.linalg.qr(v)
    s = numpy.sort(s)
    t = numpy.ones(k)*s[0]
    s = 2**(-alpha*k*(s - t)).astype(numpy.float32)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a

numpy.random.seed(1) # make results reproducible

# generate the matrix
m = 2000
n = 400
alpha = 0.05
s, u, v, a = random_svd(m, n, alpha)

# set solver options
opt = Options()
opt.verbosity = 2
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-4)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(0.5) # singular value threshold

# compute partial svd
sigma, u, v = ndarray_svd(a, opt)
print(sigma)
