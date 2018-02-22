import numpy
import sys
sys.path.append('..')

import operators
import raleigh.solver
from raleigh.ndarray_vectors import NDArrayVectors

def sort_eigenpairs(lmd, u, err_lmd, err_X):
    ind = numpy.argsort(lmd)
    w = NDArrayVectors(u)
    lmd = lmd[ind]
    err_lmd = err_lmd[ind]
    err_X = err_X[ind]
    u.copy(w, ind)
    w.copy(u)
    return lmd, u, err_lmd, err_X

numpy.random.seed(1) # to debug

opt = raleigh.solver.Options()
#opt.block_size = 5
opt.max_iter = 40
opt.res_tol = 1e-4
n = 80
#n = 160

u = NDArrayVectors(n)
v = NDArrayVectors(n)
a = numpy.asarray([i + 1 for i in range(n)])
b = 2*numpy.ones((1, n))
#p = numpy.asarray([1/(i + 1) for i in range(n)])
operatorA = operators.Diagonal(a)
operatorB = operators.Diagonal(b)
operatorP = operators.Diagonal(1/a)
opA = lambda x, y: operatorA.apply(x, y)
opB = lambda x, y: operatorB.apply(x, y)
opP = lambda x, y: operatorP.apply(x, y)
problem = raleigh.solver.Problem(v, opA, opB, 'product')
solver = raleigh.solver.Solver(problem)

solver.solve(u, opt, which = (3,3))
print('after %d iterations, %d + %d eigenvalues converged:' \
      % (solver.iteration, solver.lcon, solver.rcon))
lmd = solver.eigenvalues
lmd, u, err_lmd, err_X = sort_eigenpairs\
    (lmd, u, solver.errors_val, solver.errors_vec)
print(lmd)
lconu = solver.lcon
rconu = solver.rcon
nconu = lconu + rconu

opt.verbosity = 1
solver.solve(v, opt, which = (3,3))
print('after %d iterations, %d + %d eigenvalues converged:' \
      % (solver.iteration, solver.lcon, solver.rcon))
lmd = solver.eigenvalues
lmd, v, err_lmd, err_X = sort_eigenpairs\
    (lmd, v, solver.errors_val, solver.errors_vec)
print(lmd)

w = NDArrayVectors(u)
nconv = solver.lcon + solver.rcon
lcon = min(lconu, solver.lcon)
rcon = min(rconu, solver.rcon)
u.select(lcon)
v.select(lcon)
w.select(lcon)
opB(v, w)
q = w.dot(u)
u.mult(q, w)
v.add(w, -1.0)
opB(v, w)
sl = w.dots(v)
sl = numpy.sqrt(sl)

u.select(rcon, nconu - rcon)
v.select(rcon, nconv - rcon)
w.select(rcon, nconu - rcon)
opB(v, w)
q = w.dot(u)
u.mult(q, w)
v.add(w, -1.0)
opB(v, w)
sr = w.dots(v)
sr = numpy.sqrt(sr)
s = numpy.concatenate((sl, sr))
print('eigenvector errors:')
print('  estimated    actual')
for i in range(lcon + rcon):
    print('%e %e' % (err_X[i], s[i]))
#print(sl, sr)
#print(s)
#print(err_X)
