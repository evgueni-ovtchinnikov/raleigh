import numpy
import sys
sys.path.append('..')

import raleigh.solver
from raleigh.ndarray_vectors import NDArrayVectors

def opA(x, y):
    n = x.dimension()
    for i in range(n):
        y.data()[:,i] = (i + 1)*x.data()[:,i]

def opB(x, y):
    y.data()[:,:] = 2*x.data()[:,:]

opt = raleigh.solver.Options()
opt.block_size = 5
opt.max_iter = 23
n = 40
v = NDArrayVectors(n)
problem = raleigh.solver.Problem(v, opA, opB, 'product')
solver = raleigh.solver.Solver(problem, v, opt, (3,3))
solver.solve()
print('%d converged eigenvalues are:' % v.nvec())
print(solver.eigenvalues)
