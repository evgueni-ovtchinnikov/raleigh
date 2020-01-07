# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''Discretized Laplace operators in 1, 2 and 3 dimensions"
'''

import numpy
import scipy.sparse as scs

def lap1d(n, a):
    h = a/(n + 1)
    d = numpy.ones((n,))/(h*h)
    return scs.spdiags([-d, 2*d, -d], [-1, 0, 1], n, n, format='csr')


def lap2d(nx, ny, ax, ay):
    L = lap1d(nx, ax)
    Ly = lap1d(ny, ay)
    L = scs.csr_matrix(scs.kron(scs.eye(ny), L) + scs.kron(Ly, scs.eye(nx)))
    return L


def lap3d(nx, ny, nz, ax, ay, az):
    L = lap2d(nx, ny, ax, ay)
    Lz = lap1d(nz, az)
    L = scs.csr_matrix(scs.kron(scs.eye(nz), L) + scs.kron(Lz, scs.eye(nx*ny)))
    return L


