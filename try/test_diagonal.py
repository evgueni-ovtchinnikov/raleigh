import numpy
import sys
sys.path.append('..')

from raleigh.check_accuracy import check_eigenvectors_accuracy
import operators
import raleigh.solver
from raleigh.ndarray_vectors import NDArrayVectors

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.max_iter = 40
opt.res_tol = 1e-4
n = 80

u = NDArrayVectors(n)
v = NDArrayVectors(n)
a = numpy.asarray([i + 1 for i in range(n)])
b = 2*numpy.ones((1, n))
operatorA = operators.Diagonal(a)
operatorB = operators.Diagonal(b)
operatorP = operators.Diagonal(1/a)
opA = lambda x, y: operatorA.apply(x, y)
opB = lambda x, y: operatorB.apply(x, y)
opP = lambda x, y: operatorP.apply(x, y)
problem = raleigh.solver.Problem(v, opA, opB, 'product')
check_eigenvectors_accuracy(problem, opt, which = (3,3))
