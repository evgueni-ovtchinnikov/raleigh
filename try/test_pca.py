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
    def __init__(self):
        self.th = 1.0
        #self.nep = nep
    def set_threshold(self, th):
        self.th = th
    def satisfied(self, solver):
        if solver.rcon < 1:
            return False
        return solver.eigenvalues[solver.rcon - 1] \
            < self.th*solver.eigenvalues[0]
        #return solver.lcon + solver.rcon >= self.nep

def random_svd(m, n, alpha):
    k = min(m, n)
    u = numpy.random.randn(m, k).astype(numpy.float32)
    v = numpy.random.randn(n, k).astype(numpy.float32)
    s = numpy.random.rand(k).astype(numpy.float32)
    u, r = numpy.linalg.qr(u)
    v, r = numpy.linalg.qr(v)
    s = numpy.sort(s)
    t = numpy.ones(k)*s[0]
    s = 2**(-alpha*k*(s - t)).astype(numpy.float32)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a

numpy.random.seed(1) # make results reproducible

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.threads = 4
opt.verbosity = 2
#opt.convergence_criteria = MyConvergenceCriteria(1e-4)
opt.convergence_criteria.set_error_tolerance('eigenvector error', 1e-6)
opt.stopping_criteria = MyStoppingCriteria()
opt.stopping_criteria.set_threshold(0.9)
m = 2000
n = 400
alpha = 0.05

#a = 2*numpy.random.rand(m, n).astype(numpy.float32) - 1
#for i in range(n):
#    s = numpy.linalg.norm(a[:, i])
#    a[:, i] /= s

s, u, v, a = random_svd(m, n, alpha)

operatorA = operators.Gram(a)
opA = lambda x, y: operatorA.apply(x, y)

v = NDArrayVectors(n)
problem = raleigh.solver.Problem(v, opA)
#check_eigenvectors_accuracy(problem, opt, which = (0,-1))
check_eigenvectors_accuracy(problem, opt, which = (0,3))
