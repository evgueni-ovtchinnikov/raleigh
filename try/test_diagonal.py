import numpy
import sys
sys.path.append('..')

from raleigh.check_accuracy import check_eigenvectors_accuracy
import operators
import raleigh.solver
from raleigh.ndarray.vectors import Vectors

opt = raleigh.solver.Options()
#opt.block_size = 5
#opt.max_iter = 10
#opt.res_tol = 1e-4
opt.verbosity = 4
n = 80

u = Vectors(n)
v = Vectors(n)
#a = numpy.asarray([i + 1 for i in range(n)])
#a = numpy.asarray([1e-15*i + 1 for i in range(n)])
m = 10
a = numpy.concatenate((numpy.ones(m), 2*numpy.ones(n - m)))
b = numpy.ones(n)
#b = 2*numpy.ones((1, n))
operatorA = operators.Diagonal(a)
operatorB = operators.Diagonal(b)
operatorP = operators.Diagonal(1/a)
opA = lambda x, y: operatorA.apply(x, y)
opB = lambda x, y: operatorB.apply(x, y)
opP = lambda x, y: operatorP.apply(x, y)
problem = raleigh.solver.Problem(v, opA, opB, 'product')
check_eigenvectors_accuracy(problem, opt, which = (7,0))
#check_eigenvectors_accuracy(problem, opt, which = (3,3))
