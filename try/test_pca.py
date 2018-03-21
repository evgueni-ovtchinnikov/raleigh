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
#        lmd = solver.convergence_data('eigenvalue', i)
#        print('lambda[%d] = %e' % (i, lmd))
#        lmd_l = solver.convergence_data('eigenvalue', 0)
#        print('lambda[0] = %e' % lmd_l)
#        lmd_r = solver.convergence_data('eigenvalue', -1)
#        n = solver.convergence_data('block size')
#        print('lambda[%d] = %e' % (n - 1, lmd_r))
        err = solver.convergence_data('kinematic eigenvector error', i)
        return err >= 0 and err <= self.tol
        
class MyStoppingCriteria:
    def __init__(self, nep):
        self.nep = nep
    def satisfied(self, solver):
        return solver.lcon + solver.rcon >= self.nep

numpy.random.seed(1) # make results reproducible

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.verbosity = 2
opt.convergence_criteria = MyConvergenceCriteria(1e-4)
opt.stopping_criteria = MyStoppingCriteria(4)
m = 2000
n = 400

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
check_eigenvectors_accuracy(problem, opt, which = (0,-1))
#check_eigenvectors_accuracy(problem, opt, which = (0,3))
