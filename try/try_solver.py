import numpy
import sys
sys.path.append('..')

import operators
import raleigh.solver
from raleigh.ndarray_vectors import NDArrayVectors

numpy.random.seed(1) # to debug

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.max_iter = 40
opt.res_tol = 1e-4
n = 40
#n = 160
v = NDArrayVectors(n)
#w = v.new_orthogonal_vectors(4)
#w.select(4)
wl = v.new_random_vectors(2)
wr = v.new_random_vectors(2)
a = numpy.asarray([i + 1 for i in range(n)])
b = 2*numpy.ones((1, n))
#p = numpy.asarray([1/(i + 1) for i in range(n)])
operatorA = operators.Diagonal(a)
operatorB = operators.Diagonal(b)
operatorP = operators.Diagonal(1/a)
opA = lambda x, y: operatorA.apply(x, y)
opB = lambda x, y: operatorB.apply(x, y)
opP = lambda x, y: operatorP.apply(x, y)
problem = raleigh.solver.Problem(v, opA, opB) #, 'product')
solver = raleigh.solver.Solver(problem)
#solver.set_preconditioner(opP)
#solver.solve(v, opt, which = (3,0), extra = (0,0), init = (w, None))
solver.solve(v, opt, which = (3,3), init = (wl, wr)) #, extra = (1,1))
print('%d converged eigenvalues are:' % v.nvec())
print(solver.eigenvalues)
#solver.solve(v, opt, (2,1))
#print('%d converged eigenvalues are:' % (solver.lcon + solver.rcon))
#print(solver.eigenvalues)
