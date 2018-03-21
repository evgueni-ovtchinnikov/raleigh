import numpy
import sys
sys.path.append('..')

from raleigh.check_accuracy import check_eigenvectors_accuracy
import operators
import raleigh.solver
from raleigh.ndarray_vectors import NDArrayVectors

class MyConvergenceCriteria:
    def __init__(self, tol):
        self.tol = tol
    def satisfied(self, solver, i):
        err = solver.convergence_data('ve', i)
        return err[1] >= 0 and err[1] <= self.tol

numpy.random.seed(1) # make results reproducible

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.max_iter = 40
opt.res_tol = 1e-10
opt.verbosity = 0 #2
opt.convergence_criteria = MyConvergenceCriteria(1e-8)
m = 2000
n = 40

a = 2*numpy.random.rand(m, n).astype(numpy.float32) - 1
for i in range(n):
    s = numpy.linalg.norm(a[:, i])
    a[:, i] /= s
##d = numpy.asarray([i/5 + 1 for i in range(m)])
##a = numpy.ones((m, n))
##s = numpy.linalg.norm(a[:, 0])
##a[:, 0] /= s
##for i in range(1, n):
##    a[:, i] = d*a[:, i - 1]
##    s = numpy.linalg.norm(a[:, i])
##    a[:, i] /= s
operatorA = operators.Gram(a)
opA = lambda x, y: operatorA.apply(x, y)

v = NDArrayVectors(n)
problem = raleigh.solver.Problem(v, opA)
check_eigenvectors_accuracy(problem, opt, which = (0,3))
