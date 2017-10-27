import numpy
import sys
sys.path.append('..')

from raleigh.ndarray_vectors import NDArrayVectors
from raleigh.solver import *

def opA(x, y):
    n = x.dimension()
    for i in range(n):
        y.data()[i,:] = (i + 1)*x.data()[i,:]

def opB(x, y):
    y.data()[:,:] = 2*x.data()[:,:]
##    for i in range(n):
##        y[i,:] = 2*x[i,:]

opt = Options()
opt.block_size = 2
n = 8
v = NDArrayVectors(numpy.ndarray((n,1)))
problem = Problem(v, opA, opB, 'product')
#problem = Problem(n, opA)
solver = Solver(problem, opt)
solver.solve()
