# -*- coding: utf-8 -*-
""" Kernel PCA via full SVD

Usage:
  try_svd_kpca [--help | -h | options] <data> <nim>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)
  nim   number of images to use (< 0 : all)

Options:
  -a <alpha>, --alpha=<alpha>  H1 term factor [default: 0]
  -b <beta> , --beta =<beta>   L2 term factor [default: 1]
  -s <stncl>, --stncl=<stncl>  Laplace discretization stencil pts [default: 5]

Created on Tue Oct 30 09:30:40 2018

@author: Evgueni Ovtchinnikov, UKRI
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<data>']
ni = int(args['<nim>'])
alpha = float(args['--alpha'])
beta = float(args['--beta'])
stncl = int(args['--stncl'])

import numpy
import numpy.linalg as nla
import pylab
import scipy.linalg as sla
import sys

raleigh_path = '..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.algebra import Vectors, Matrix

class Laplace2D:
    def __init__(self, sx, sy, nx, ny, alpha = 0.0, beta = 1.0, stencil = 5):
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.beta = beta
        self.stencil = stencil
        self.hx = sx/(nx + 1)
        self.hy = sy/(ny + 1)
        self.xh = 1/(self.hx * self.hx)
        self.yh = 1/(self.hy * self.hy)
    def apply(self, u, v):
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

print('loading images...')
all_images = numpy.load(file)

m_all, ny, nx = all_images.shape
dt = all_images.dtype.type
n = nx*ny

if ni < 0 or ni > m_all:
    ni = m_all

if ni < m_all:
    print('using first %d images only...' % ni)
    m = ni
    A = all_images[:m,:,:]
else:
    m = m_all
    A = all_images

A = numpy.reshape(A, (m, n))
print('computing svd...')
AAT = numpy.dot(A, A.T)
lmd, uA = sla.eigh(AAT)
#uA, sA, vAt = sla.svd(A, full_matrices = False)
#sA = numpy.sqrt(abs(lmd))[::-1]
#print(sA)
uA = uA[:, ::-1]
vA = numpy.dot(uA.T, A)
sA = nla.norm(vA, axis = 1)
eigim = (vA.T/sA).T
coord = uA*sA
D = A - numpy.dot(coord, eigim)
print(nla.norm(D))
err = numpy.ndarray((m, m))
#err[:,0] = nla.norm(coord, axis = 1)
err = coord*coord
for j in range(m - 1):
    k = m - 1 - j
    err[:, k - 1] += err[:,k]
err = numpy.sqrt(err)
#print(err[:,0])

K = Laplace2D(ny/nx, 1.0, ny, nx, alpha, beta, stncl)
KA = numpy.ndarray((m, n), dtype = dt)
K.apply(A, KA)
#if (alpha != 0.0 or beta != 1.0):
print('normalizing images...')
v = Vectors(A)
w = Vectors(KA)
s = numpy.sqrt(abs(w.dots(v)))
#print(s)
v.scale(s)
w.scale(s)
print('computing kernel svd...')
B = w.dot(v)
lmd, uB = sla.eigh(B)
uB = uB[:, ::-1]
vB = numpy.dot(uB.T, A)
w = numpy.ndarray((m, n), dtype = dt)
K.apply(vB, w)
v = Vectors(vB)
w = Vectors(w)
sB = numpy.sqrt(abs(w.dots(v)))
eigimK = (vB.T/sB).T
coordK = uB*sB
D = A - numpy.dot(coordK, eigimK)
print(nla.norm(D))
#v = Vectors(A)
#w = Vectors(KA)
#s = numpy.sqrt(abs(w.dots(v)))
#print(s)
errK = numpy.ndarray((m, m))
#errK[:,0] = nla.norm(coordK, axis = 1)
errK = coordK*coordK
for j in range(m - 1):
    k = m - 1 - j
    errK[:, k - 1] += errK[:,k]
errK = numpy.sqrt(errK)
#print(errK[:,0])
##sB = numpy.sqrt(abs(lmd))[::-1]
##print(sB)
#
pylab.figure()
pylab.plot(numpy.arange(1, m + 1, 1), sA)
pylab.xscale('log')
pylab.yscale('log')
pylab.grid()
pylab.title('singular values')
pylab.plot(numpy.arange(1, m + 1, 1), sB)
pylab.xscale('log')
pylab.yscale('log')
pylab.grid()
pylab.title('kernel singular values')
pylab.show()

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= ni:
        break
    pylab.figure()
    pylab.plot(numpy.arange(1, m + 1, 1), err[i,:])
    pylab.plot(numpy.arange(1, m + 1, 1), errK[i,:])
    pylab.yscale('log')
    pylab.grid()
    pylab.title('PCA error')
    pylab.show()
    while True:
        j = int(input('number of PCs (negative to exit): '))
        if j <= 0 or j > m:
            break
#        D = A - numpy.dot(coordK[:, : j], eigimK[: j, :])
#        K.apply(D, KA)
#        v = Vectors(D)
#        w = Vectors(KA)
#        s = numpy.sqrt(abs(w.dots(v)))
        img = numpy.dot(coord[i, : j], eigim[: j, :])
        pylab.figure()
        pylab.title('image %d, %d PCs' % (i, j))
        pylab.imshow(numpy.reshape(img, (ny, nx)), cmap = 'gray')
        img = numpy.dot(coordK[i, : j], eigimK[: j, :])
        d = numpy.reshape(img - A[i,:], (1, n))
        Kd = d.copy()
        K.apply(d, Kd)
        v = Vectors(d)
        w = Vectors(Kd)
        s = numpy.sqrt(abs(w.dots(v)))
        pylab.figure()
        pylab.title('image %d, %d kernel PCs' % (i, j))
        pylab.imshow(numpy.reshape(img, (ny, nx)), cmap = 'gray')
        print('errors: %.1e %.1e %.1e' % (err[i, j], errK[i,j], s))
        pylab.show()

print('done')