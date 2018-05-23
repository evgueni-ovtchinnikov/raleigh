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

try:
    from raleigh.ndarray.mkl import mkl, Cblas
    HAVE_MKL = True
except:
    HAVE_MKL = False

class SVD:
    def __init__(self, a):
        self.__a = a
        self.__type = type(a[0,0])
    def apply(self, x, y):
        u = x.data()
        v = y.data()
        type_x = type(u[0,0])
        mixed_types = type_x is not self.__type
        if mixed_types:
            z = numpy.dot(u.astype(self.__type), self.__a.T)
            v[:,:] = numpy.dot(z, self.__a).astype(type_x)
        else:
#            if HAVE_MKL:
#                m, na = self.__a.shape
#                n, k = u.shape
#                if n != na:
#                    raise ValueError('mismatching dimensions %d != %d' % (n, na))
#                mkl_n = ctypes.c_int(n)
#                mkl_m = ctypes.c_int(m)
#                mkl_k = ctypes.c_int(k)
#            else:
            z = numpy.dot(u, self.__a.T)
            v[:,:] = numpy.dot(z, self.__a)
