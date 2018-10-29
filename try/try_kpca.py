# -*- coding: utf-8 -*-
""" Kernel PCA demo

Usage:
  ipca_images [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -n <nim>, --nimgs=<nim>   number of images to use (< 0 : all) [default: -1]
  -a <alp>, --alpha=<alp>   H1 term factor [default: 0]

Created on Mon Oct 29 11:11:41 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<data>']
ni = int(args['--nimgs'])
alpha = float(args['--alpha'])

import numpy
import pylab
import scipy.linalg as sla
import sys

raleigh_path = '..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Problem, Solver, Options
from raleigh.algebra import Vectors, Matrix

class Laplace2D:
    def __init__(self, sx, sy, nx, ny, alpha = 0.0):
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
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
        xh = self.xh*alpha
        yh = self.yh*alpha
        mx = nx - 1
        my = ny - 1
        n = nx*ny
        u = numpy.reshape(u, (m, nx, ny))
        v = numpy.reshape(v, (m, nx, ny))
        d = 2*(xh + yh) + 1.0
        v[:, :, :] = d*u
        v[:, 1 : nx, :] -= xh*u[:, 0 : mx, :]
        v[:, 0 : mx, :] -= xh*u[:, 1 : nx, :]
        v[:, :, 1 : ny] -= yh*u[:, :, 0 : my]
        v[:, :, 0 : my] -= yh*u[:, :, 1 : ny]
        u = numpy.reshape(u, (m, n))
        v = numpy.reshape(v, (m, n))

class OperatorSVD:
    def __init__(self, op, kernel):
        self.op = op
        self.kernel = kernel
    def apply(self, x, y):
        m, n = self.op.shape()
        k = x.nvec()
        z = Vectors(n, k, x.data_type())
        w = Vectors(n, k, x.data_type())
        self.op.apply(x, z, transp = True)
        self.kernel.apply(z, w)
        self.op.apply(w, y)

class MyStoppingCriteria:
    def __init__(self):
        self.ncon = 0
        self.iteration = 0
    def satisfied(self, solver):
        if solver.rcon <= self.ncon:
            return False
        print('iterations: %d, eigenimages computed: %d' % \
            (solver.iteration, solver.rcon))
        lmd = solver.eigenvalues[self.ncon : solver.rcon]
        sigma = -numpy.sort(-numpy.sqrt(lmd))
        if self.ncon == 0:
            self.sigma = sigma[0]
        for i in range(solver.rcon - self.ncon):
            print('sigma[%4d] = %14f = %10f*sigma[0]' % \
            (self.ncon + i, sigma[i], sigma[i]/self.sigma))
        self.ncon = solver.rcon
        done = (input('more? ') == 'n')
        self.iteration = solver.iteration
        return done

numpy.random.seed(1) # make results reproducible

print('loading images...')
all_images = numpy.load(file)

m_all, ny, nx = all_images.shape
n = nx*ny

if ni < 0 or ni > m_all:
    ni = m_all

if ni < m_all:
    print('using first %d images only...' % ni)
    m = ni
    images = all_images[:m,:,:]
else:
    m = m_all
    images = all_images

dt = images.dtype.type

b = max(1, min(m, n)//100)
block_size = 128 #32
while block_size <= b - 16:
    block_size += 32
print('using block size %d' % block_size)

opt = Options()
opt.block_size = block_size
opt.max_iter = 500
opt.stopping_criteria = MyStoppingCriteria()
opt.verbosity = -1

images = numpy.reshape(images, (m, n))
op = Matrix(images)
kernel = Laplace2D(ny/nx, 1.0, ny, nx, alpha)
if (alpha != 0):
    print('normalizing images...')
    v = Vectors(images)
    w = Vectors(numpy.ndarray((m, n), dtype = dt))
    kernel.apply(v, w)
    s = numpy.sqrt(abs(w.dots(v)))
    v.scale(s)

opSVD = OperatorSVD(op, kernel)
v = Vectors(m, data_type = dt)
problem = Problem(v, lambda x, y: opSVD.apply(x, y))
solver = Solver(problem)

solver.solve(v, opt, which = (0, -1))

ncon = v.nvec()
iterations = opt.stopping_criteria.iteration
print('%d eigenimages computed in %d iterations' % (ncon, iterations))

u = Vectors(n, ncon, v.data_type())
op.apply(v, u, transp = True)
vv = v.dot(v)
if alpha != 0:
    w = u.new_vectors(ncon)
    kernel.apply(u, w)
    uu = -u.dot(w)
else:
    uu = -u.dot(u)
lmd, x = sla.eigh(uu, vv, turbo = False)
w = v.new_vectors(ncon)
v.multiply(x, w)
w.copy(v)
w = u.new_vectors(ncon)
u.multiply(x, w)
w.copy(u)
if alpha != 0:
    kernel.apply(u, w)
    sigma = numpy.sqrt(abs(u.dots(w)))
else:
    sigma = numpy.sqrt(abs(u.dots(u)))
u.scale(sigma)
v.scale(sigma, multiply = True)

eigim = numpy.reshape(u.data(), (ncon, ny, nx))
coord = v.data()
numpy.save('eigim.npy', eigim)
numpy.save('coord.npy', coord)
numpy.save('sigma.npy', sigma[:ncon])

images -= numpy.dot(v.data().T, u.data())
if (alpha != 0):
    print('computing errors...')
    v = Vectors(images)
    w = Vectors(numpy.ndarray((m, n), dtype = dt))
    kernel.apply(v, w)
    s = numpy.sqrt(abs(w.dots(v)))
else:
    s = numpy.linalg.norm(images, axis = 1)
print(numpy.amax(s))
#print(s.shape)

#while True:
#    i = int(input('image number (negative to exit): '))
#    if i < 0 or i >= ncon:
#        break
#    #image = numpy.reshape(u.data()[i,:], (ny, nx))
#    image = eigim[i,:,:]
#    pylab.figure()
#    pylab.title('eigenface %d' % i)
#    pylab.imshow(image, cmap = 'gray')
#    pylab.show()
