import numpy
import sys
sys.path.append('..')

from raleigh.ndarray_vectors_opt import NDArrayVectors
from raleigh.solver import *

def opA(x, y):
    n = x.dimension()
    for i in range(n):
        y.data()[:,i] = (i + 1)*x.data()[:,i]
#        y.data()[i,:] = (i + 1)*x.data()[i,:]

def opB(x, y):
    y.data()[:,:] = 2*x.data()[:,:]
##    for i in range(n):
##        y[i,:] = 2*x[i,:]

opt = Options()
opt.block_size = 5
n = 20
v = NDArrayVectors(numpy.ndarray((1,n), order = 'C'))
problem = Problem(v, opA, opB, 'product')
#problem = Problem(n, opA)
solver = Solver(problem, opt, (6,1))
solver.solve()
