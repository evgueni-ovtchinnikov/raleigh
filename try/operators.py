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
        self.type = type(a[0,0])
    def apply(self, x, y):
        u = x.data()
        type_x = type(u[0,0])
        if type_x is not self.type:
            mixed_types = True
            u = u.astype(self.type)
        else:
            mixed_types = False
        z = numpy.dot(u, self.a.T)
        #z = numpy.dot(x.data(), self.a.T)
        if mixed_types:
            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
        else:
            y.data()[:,:] = numpy.dot(z, self.a)
