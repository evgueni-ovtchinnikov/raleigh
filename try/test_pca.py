import numpy
import sys
sys.path.append('..')

from raleigh.check_accuracy import check_eigenvectors_accuracy
#from raleigh.ndarray_svd import partial_svd
import operators
import raleigh.solver
from raleigh.ndarray.vectors import Vectors
from random_matrix_for_svd import random_matrix_for_svd

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
    def __init__(self):
        self.th = 1.0
        #self.nep = nep
    def set_threshold(self, th):
        self.th = th*th
    def satisfied(self, solver):
        if solver.rcon < 1:
            return False
        return numpy.amin(solver.eigenvalues) < self.th*solver.eigenvalues[0]
#        print(solver.eigenvalues[solver.rcon - 1]/solver.eigenvalues[0])
#        return solver.eigenvalues[solver.rcon - 1] \
#            < self.th*solver.eigenvalues[0]
        #return solver.lcon + solver.rcon >= self.nep

numpy.random.seed(1) # make results reproducible

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.threads = 4
opt.verbosity = 2
#opt.convergence_criteria = MyConvergenceCriteria(1e-4)
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-8)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(0.01)
m = 10000
n = 4000
k = 200
alpha = 0.05

dt = numpy.float32

a = 2*numpy.random.rand(m, n).astype(dt) - 1
s = numpy.linalg.norm(a, axis = 0)
a /= s
#for i in range(n):
#    s = numpy.linalg.norm(a[:, i])
#    a[:, i] /= s

#s, u, v, a = random_svd(m, n, alpha)

alpha = 0.01
sigma = lambda t: 2**(-alpha*t*t).astype(dt)
s, u, v, b = random_matrix_for_svd(m, n, k, sigma, dt)

a = 1e-3*a + b

#sigma, u, v = partial_svd(a, opt)
#print(sigma)

operatorATA = operators.SVD(a)
op = lambda x, y: operatorATA.apply(x, y)

v = Vectors(n)
problem = raleigh.solver.Problem(v, op)
check_eigenvectors_accuracy(problem, opt, which = (0,-1))
#check_eigenvectors_accuracy(problem, opt, which = (0,3))
