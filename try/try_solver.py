import numpy
import sys
sys.path.append('..')

import operators
import raleigh.solver
from raleigh.algebra import Vectors, Matrix
#from raleigh.ndarray.cblas_vectors import Vectors
#from raleigh.ndarray.numpy_vectors import Vectors
#from raleigh.ndarray.vectors import Vectors
import scipy.linalg as sla

numpy.random.seed(1) # to debug

opt = raleigh.solver.Options()
opt.block_size = 2
#opt.max_iter = 16
opt.res_tol = 1e-4
#opt.verbosity = 2 #3
n = 40
#n = 160
v = Vectors(n, data_type = numpy.complex128)
#w = v.new_orthogonal_vectors(4)
#w.select(4)
wl = v.new_vectors(2)
wr = v.new_vectors(2)
#wl.fill_random()
#wr.fill_random()
a = numpy.asarray([i + 1 for i in range(n)])
b = 2*numpy.ones((1, n))
#p = numpy.asarray([1/(i + 1) for i in range(n)])
#operatorA = operators.Diagonal(a)
operatorA = operators.CentralDiff(n)
operatorB = operators.Diagonal(b)
operatorP = operators.Diagonal(1/a)
opA = lambda x, y: operatorA.apply(x, y)
opB = lambda x, y: operatorB.apply(x, y)
opP = lambda x, y: operatorP.apply(x, y)
#u = v.new_vectors(1)
#w = v.new_vectors(1)
#u.fill(1.0)
#print(u.data())
#opA(u, w)
#print(w.data())
problem = raleigh.solver.Problem(v, opA, opB, 'product')
solver = raleigh.solver.Solver(problem)
#solver.set_preconditioner(opP)
#solver.solve(v, opt, which = (3,0), extra = (0,0), init = (w, None))
#solver.solve(v, opt, which = (3,3), init = (wl, wr)) #, extra = (1,1))
solver.solve(v, opt, which = (3,3))
print('after %d iterations, %d converged eigenvalues are:' \
      % (solver.iteration, v.nvec()))
print(solver.eigenvalues)
#solver.solve(v, opt, (2,1))
#print('%d converged eigenvalues are:' % (solver.lcon + solver.rcon))
#print(solver.eigenvalues)

d = 1j*numpy.ones((n - 1,))
A = numpy.diag(d, 1) - numpy.diag(d, -1)
B = numpy.diag(2*numpy.ones((n,)))
A = numpy.dot(B, numpy.dot(A, B))
lmd, X = sla.eigh(A, B)
print(lmd)

#x = v.new_vectors(2)
#y = v.new_vectors(2)
#x.fill_orthogonal()
#print(x.data())
#x.apply(A, y)
#print(y.data())

v = Vectors(2, data_type = numpy.complex128)
B = Matrix(A[:3,:2].copy())
C = Matrix(A[:2,:3].copy())
print('B:')
print(B.data())
x = v.new_vectors(2)
y = Vectors(3, 2, data_type = numpy.complex128)
z = y.new_vectors(2)
x.fill_orthogonal()
print('x:')
print(x.data())
x.apply(B, y)
print('y = B x:')
print(y.data())
y.apply(B, x, transp = True)
print('x = B* y:')
print(x.data())
C = C.data()
print('C:')
print(C)
y.multiply(C, z)
print('z:')
print(z.data())
D = C.T.copy()
#z.add(y, -1, D.T)
z.add(y, -1, C)
print(z.data())
