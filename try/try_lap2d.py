# -*- coding: utf-8 -*-
""" Laplace 2D demo

Usage:
  svds [--help | -h | options] <sx> <sy> <nx> <ny>

Arguments:
  sx  size in x
  sy  size in y
  nx  number of grid points in x
  ny  number of grid points in y

Created on Mon Oct 29 09:41:49 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

import numpy
import sys

raleigh_path = '..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Problem, Solver, Options
from raleigh.algebra import Vectors, Matrix

numpy.random.seed(1) # to make results reproducible

sx = int(args['<sx>'])
sy = int(args['<sy>'])
nx = int(args['<nx>'])
ny = int(args['<ny>'])

class Laplace2D:
    def __init__(self, sx, sy, nx, ny):
        self.nx = nx
        self.ny = ny
        self.hx = sx/(nx + 1)
        self.hy = sy/(ny + 1)
        self.xh = 1/(self.hx*self.hx)
        self.yh = 1/(self.hy*self.hy)
    def apply(self, vu, vv):
        u = vu.data()
        v = vv.data()
        m = u.shape[0]
        nx = self.nx
        ny = self.ny
        xh = self.xh
        yh = self.yh
        mx = nx - 1
        my = ny - 1
        n = nx*ny
        u = numpy.reshape(u, (m, nx, ny))
        v = numpy.reshape(v, (m, nx, ny))
        d = 2*(xh + yh)
        v[:, :, :] = d*u
        v[:, 1 : nx, :] -= xh*u[:, 0 : mx, :]
        v[:, 0 : mx, :] -= xh*u[:, 1 : nx, :]
        v[:, :, 1 : ny] -= yh*u[:, :, 0 : my]
        v[:, :, 0 : my] -= yh*u[:, :, 1 : ny]
        u = numpy.reshape(u, (m, n))
        v = numpy.reshape(v, (m, n))

n = nx*ny
L = Laplace2D(sx, sy, nx, ny)
I = Vectors(numpy.eye(n))
M = Vectors(numpy.ndarray((n, n)))
L.apply(I, M)
#print(M.data())
v = Vectors(n)
opA = lambda x, y: L.apply(x, y)
problem = Problem(v, opA)
solver = Solver(problem)
opt = Options()
#opt.block_size = 4
#opt.verbosity = 1
nep = 4
solver.solve(v, opt, which = (nep,0))
print('after %d iterations, %d converged eigenvalues are:' \
      % (solver.iteration, v.nvec()))
print(solver.eigenvalues)
u = v.data()
for i in range(nep):
    print (numpy.reshape(u[i,:], (nx, ny)))
