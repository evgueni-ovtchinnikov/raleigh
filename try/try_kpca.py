# -*- coding: utf-8 -*-
""" Kernel PCA demo

Usage:
  try_kpca [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -n <nim>  , --nimgs=<nim>    number of images to use (<0: all) [default: -1]
  -a <alpha>, --alpha=<alpha>  H1 term factor [default: 0]
  -b <beta> , --beta =<beta>   L2 term factor [default: 1]
  -t <sigm> , --thres=<sigm>   sigma threshold (rel) [default: 0.01]
  -s <stncl>, --stncl=<stncl>  Laplace discretization stencil pts [default: 5]

Created on Mon Oct 29 11:11:41 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)
print(args)

file = args['<data>']
#ni = int(args['--nimgs'])
ni = int(args['-n'])
alpha = float(args['--alpha'])
beta = float(args['--beta'])
th = float(args['--thres'])
stncl = int(args['--stncl'])

import numpy
import numpy.linalg as nla
import pylab
import scipy.linalg as sla
import sys

raleigh_path = '..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Problem, Solver, Options
from raleigh.algebra import Vectors, Matrix

class Laplace2D:
    def __init__(self, sx, sy, nx, ny, alpha = 1.0, beta = 0.0, stencil = 5):
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.beta = beta
        self.stencil = stencil
        self.hx = sx/(nx + 1)
        self.hy = sy/(ny + 1)
        self.xh = 1/(self.hx * self.hx)
        self.yh = 1/(self.hy * self.hy)
    def apply(self, vu, vv):
        u = vu.data()
        v = vv.data()
        m = u.shape[0]
        nx = self.nx
        ny = self.ny
        xh = self.xh * self.alpha
        yh = self.yh * self.alpha
        mx = nx - 1
        my = ny - 1
        n = nx*ny
        u = numpy.reshape(u, (m, nx, ny))
        v = numpy.reshape(v, (m, nx, ny))
        if self.stencil == 5:
            d = 2*(xh + yh) + self.beta
            v[:, :, :] = d*u
            v[:, 1 : nx, :] -= xh*u[:, 0 : mx, :]
            v[:, 0 : mx, :] -= xh*u[:, 1 : nx, :]
            v[:, :, 1 : ny] -= yh*u[:, :, 0 : my]
            v[:, :, 0 : my] -= yh*u[:, :, 1 : ny]
        else:
            xxh = (yh - 2*xh)/3
            yyh = (xh - 2*yh)/3
            xyh = -(xh + yh)/6
            d = (xh + yh)*4/3 + self.beta
            v[:, :, :] = d*u
            v[:, 1 : nx, :] += xxh*u[:, 0 : mx, :]
            v[:, 0 : mx, :] += xxh*u[:, 1 : nx, :]
            v[:, :, 1 : ny] += yyh*u[:, :, 0 : my]
            v[:, :, 0 : my] += yyh*u[:, :, 1 : ny]
            v[:, 1 : nx, 1 : ny] += xyh*u[:, 0 : mx, 0 : my]
            v[:, 0 : mx, 1 : ny] += xyh*u[:, 1 : nx, 0 : my]
            v[:, 1 : nx, 0 : my] += xyh*u[:, 0 : mx, 1 : ny]
            v[:, 0 : mx, 0 : my] += xyh*u[:, 1 : nx, 1 : ny]
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
        self.sigma = 1
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
        done = sigma[-1] <= th*self.sigma
        #done = (input('more? ') == 'n')
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
#opt.convergence_criteria.set_error_tolerance \
#    ('kinematic eigenvector error', 1e-3)
#    ('residual', 1e-4)
opt.stopping_criteria = MyStoppingCriteria()
#opt.detect_stagnation = False
opt.verbosity = -1

images = numpy.reshape(images, (m, n))
print(alpha, beta)
K = Laplace2D(ny/nx, 1.0, ny, nx, alpha, beta, stncl)
print('normalizing images...')
A = images
KA = numpy.ndarray((m, n), dtype = dt)
v = Vectors(A)
w = Vectors(KA)
K.apply(v, w)
#if (alpha != 0.0 or beta != 1.0):
s = numpy.sqrt(abs(w.dots(v)))
#if alpha != 0:
#    print('normalizing images...')
#    v = Vectors(images)
#    w = Vectors(numpy.ndarray((m, n), dtype = dt))
#    K.apply(v, w)
#    s = numpy.sqrt(abs(w.dots(v)))
#    v.scale(s)
v.scale(s)
w.scale(s)

print('computing kernel svd...')
B = w.dot(v)
lmd, uB = sla.eigh(B)
uB = uB[:, ::-1]
vB = numpy.dot(uB.T, A)
w = numpy.ndarray((m, n), dtype = dt)
v = Vectors(vB)
w = Vectors(w)
K.apply(v, w)
sB = numpy.sqrt(abs(w.dots(v)))
eigimK = (vB.T/sB).T
coordK = uB*sB
D = A - numpy.dot(coordK, eigimK)
print(nla.norm(D))

op = Matrix(images)
opSVD = OperatorSVD(op, K)
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
    K.apply(u, w)
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
    K.apply(u, w)
    sigma = numpy.sqrt(abs(u.dots(w)))
else:
    sigma = numpy.sqrt(abs(u.dots(u)))
#print(sigma)
    
#cB = uB[:, :ncon].T.copy()
#print(cB.shape)
#x = Vectors(cB)
#y = v.clone()
#q = x.dot(v)
#v.multiply(q, y)
#y.add(x, -1.0)
#err = numpy.sqrt(y.dots(y))
##print(err)
#print('max coordinate error: %.1e' % numpy.amax(err))

u.scale(sigma)
v.scale(sigma, multiply = True)

eigim = numpy.reshape(u.data(), (ncon, ny, nx))
coord = v.data()
numpy.save('eigim.npy', eigim)
numpy.save('coord.npy', coord)
numpy.save('sigma.npy', sigma[:ncon])

#print(sB[:ncon])
x = u.new_vectors(ncon)
y = u.new_vectors(ncon)
z = Vectors(eigimK[:ncon,:])
K.apply(z, y)
q = y.dot(u)
u.multiply(q, y)
y.add(z, -1.0)
K.apply(y, x)
err = numpy.sqrt(x.dots(y))
#print(err)
print('max eigenimage error: %.1e' % numpy.amax(err))

#x = v.new_vectors(ncon)
##y = v.new_vectors(ncon)
#y = v.clone()
#c = coordK[:, :ncon].T.copy()
#z = Vectors(c)
#y.scale(sigma)
##z.scale(sigma)
##print(numpy.sqrt(v.dots(v)))
##print(numpy.sqrt(z.dots(z)))
#q = z.dot(y)
#y.multiply(q, x)
#x.add(z, -1.0)
##v.copy(y)
##y.add(z, -1.0)
#err = numpy.sqrt(x.dots(x))
#print(err)
#print('max coordinate error: %.1e' % numpy.amax(err))

#t = numpy.linalg.norm(images, axis = 1)
#images -= numpy.dot(v.data().T, u.data())
#s = numpy.linalg.norm(images, axis = 1)
#print(numpy.amax(s/t))
#if (alpha != 0):
#    print('computing errors...')
#    v = Vectors(images)
#    w = Vectors(numpy.ndarray((m, n), dtype = dt))
#    K.apply(v, w)
#    #r = numpy.sqrt(abs(v.dots(v)))
#    err = numpy.sqrt(abs(w.dots(v)))
#    #t = numpy.sqrt(abs(w.dots(w)))
#    #print(numpy.amax(r))
#    #print(numpy.amax(t))
#else:
#    err = numpy.linalg.norm(images, axis = 1)
#print(numpy.amax(err))
##print(s.shape)

eigim = numpy.reshape(eigim, (ncon, n))
while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= m:
        break
    img = numpy.dot(eigim.T, coord[:, i])
    d = numpy.reshape(img - images[i,:], (1, n))
    Kd = d.copy()
    v = Vectors(d)
    w = Vectors(Kd)
    K.apply(v, w)
    s = numpy.sqrt(abs(w.dots(v)))
    print('error: %.1e' % s)
    pylab.figure()
    pylab.title('image %d' % i)
    pylab.imshow(numpy.reshape(img, (ny, nx)), cmap = 'gray')
    img = numpy.dot(coordK[i, : ncon], eigimK[: ncon, :])
    d = numpy.reshape(img - images[i,:], (1, n))
    Kd = d.copy()
    v = Vectors(d)
    w = Vectors(Kd)
    K.apply(v, w)
    s = numpy.sqrt(abs(w.dots(v)))
    print('error: %.1e' % s)
    pylab.figure()
    pylab.title('image %d' % i)
    pylab.imshow(numpy.reshape(img, (ny, nx)), cmap = 'gray')
#    #image = numpy.reshape(u.data()[i,:], (ny, nx))
#    image = eigim[i,:,:]
#    pylab.figure()
#    pylab.title('eigenface %d' % i)
#    pylab.imshow(image, cmap = 'gray')
    pylab.show()

print('done')