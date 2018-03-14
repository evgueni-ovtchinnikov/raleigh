'''Self-adjoint operators library
'''

import numpy

class Diagonal:
    def __init__(self, d):
        self.d = d
    def apply(self, x, y):
        y.data()[:,:] = self.d * x.data()

class Gram:
    def __init__(self, a):
        self.a = a
    def apply(self, x, y):
        z = numpy.dot(x.data(), self.a.T)
        y.data()[:,:] = numpy.dot(z, self.a)
