'''Self-adjoint operators library
'''

import sys

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.algebra import Vectors, Matrix

class Diagonal:
    def __init__(self, d):
        self.d = d
    def apply(self, x, y):
        y.data()[:,:] = self.d * x.data()

class CentralDiff:
    def __init__(self, n):
        self.n = n
    def apply(self, x, y):
        y.data()[:,0] = 1j * x.data()[:,1]
        y.data()[:,-1] = -1j * x.data()[:,-2]
        y.data()[:, 1 : -1] = 1j * x.data()[:, 2:] - 1j * x.data()[:, :-2]

class SVD:
    def __init__(self, array):
        self.matrix = Matrix(array)
    def apply(self, x, y, transp = False):
        m, n = self.matrix.shape()
        k = x.nvec()
        if transp:
            z = Vectors(n, k, x.data_type())
            self.matrix.apply(x, z, transp = True)
            self.matrix.apply(z, y)
        else:
            z = Vectors(m, k, x.data_type())
            self.matrix.apply(x, z)
            self.matrix.apply(z, y, transp = True)
