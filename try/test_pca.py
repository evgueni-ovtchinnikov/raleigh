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
        return numpy.amin(solver.eigenvalues) < \
                self.th*numpy.amax(solver.eigenvalues)
        #return solver.lcon + solver.rcon >= self.nep

numpy.random.seed(1) # make results reproducible

opt = raleigh.solver.Options()
opt.block_size = 64 #192
opt.threads = 4
opt.verbosity = 2 #1 #2
opt.max_iter = 13 #200
#opt.convergence_criteria = MyConvergenceCriteria(1e-4)
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-25) #1e-8)
#opt.convergence_criteria.set_error_tolerance('residual', 1e-6) #1e-8)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(0.01)
#opt.detect_stagnation = False

m = 10000
n = 4000
k = 200
#alpha = 0.05
EXP = 1

dt_op = numpy.float32
dt_v = numpy.float32

a = 2*numpy.random.rand(m, n).astype(dt_op) - 1
s = numpy.linalg.norm(a, axis = 0)
a /= s
#for i in range(n):
#    s = numpy.linalg.norm(a[:, i])
#    a[:, i] /= s

#s, u, v, a = random_svd(m, n, alpha)

#alpha = 0.01
#sigma = lambda t: 2**(-alpha*t*t).astype(dt)
if EXP == 1:
    alpha = 0.05
    f_sigma = lambda t: 2**(-alpha*t).astype(dt_op)
else:
    alpha = 0.01
    f_sigma = lambda t: 2**(-alpha*t*t).astype(dt_op)
s, u, v, b = random_matrix_for_svd(m, n, k, f_sigma, dt_op)
v0 = Vectors(v[:,::-1].T)

eps = 0 #.001
a = eps*a + b

#sigma, u, v = partial_svd(a, opt)
#print(sigma)

operatorATA = operators.SVD(a)
op = lambda x, y: operatorATA.apply(x, y)

v = Vectors(n, data_type = dt_v, with_mkl = False)
problem = raleigh.solver.Problem(v, op)
if eps == 0:
    check_eigenvectors_accuracy(problem, opt, which = (0,-1), v_ex = v0)
else:
    check_eigenvectors_accuracy(problem, opt, which = (0,-1))
#check_eigenvectors_accuracy(problem, opt, which = (0,3))
