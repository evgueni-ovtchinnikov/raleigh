'''Self-adjoint operators library
'''

import numpy

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
    def __init__(self, a):
        self.a = a
        self.type = type(a[0,0])
    def apply(self, x, y):
        u = x.data()
        type_x = type(u[0,0])
        mixed_types = type_x is not self.type
        if mixed_types:
            z = numpy.dot(u.astype(self.type), self.a.T)
            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
        else:
            z = numpy.dot(u, self.a.T)
            y.data()[:,:] = numpy.dot(z, self.a)
