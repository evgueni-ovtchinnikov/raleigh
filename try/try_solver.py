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

def opP(x, y):
    n = x.dimension()
    for i in range(n):
        y.data()[:,i] = x.data()[:,i]/(i + 1)

opt = raleigh.solver.Options()
opt.block_size = 5
opt.max_iter = 30
opt.res_tol = 1e-4
n = 160
v = NDArrayVectors(n)
problem = raleigh.solver.Problem(v, opA, opB, 'product')
solver = raleigh.solver.Solver(problem)
solver.set_preconditioner(opP)
solver.solve(v, opt, (3,0))
#solver.solve(v, opt, (3,3))
print('%d converged eigenvalues are:' % v.nvec())
print(solver.eigenvalues)
#solver.solve(v, opt, (2,1))
#print('%d converged eigenvalues are:' % (solver.lcon + solver.rcon))
#print(solver.eigenvalues)
