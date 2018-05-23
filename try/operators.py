'''Self-adjoint operators library
'''

import ctypes
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
            if HAVE_MKL:
                cblas = Cblas(self.__type)
                m, na = self.__a.shape
                k, n = u.shape
                if n != na:
                    raise ValueError('mismatching dimensions %d != %d' % (n, na))
                z = numpy.ndarray((k, m), dtype = self.__type)
                mkl_n = ctypes.c_int(n)
                mkl_m = ctypes.c_int(m)
                mkl_k = ctypes.c_int(k)
                data_u = u.ctypes.data
                data_v = v.ctypes.data
                data_a = self.__a.ctypes.data
                ptr_u = ctypes.c_void_p(data_u)
                ptr_v = ctypes.c_void_p(data_v)
                ptr_a = ctypes.c_void_p(data_a)
                ptr_z = ctypes.c_void_p(z.ctypes.data)
                cblas.gemm(Cblas.ColMajor, Cblas.Trans, Cblas.NoTrans, \
                    mkl_m, mkl_k, mkl_n, \
                    cblas.mkl_one, ptr_a, mkl_n, ptr_u, mkl_n, \
                    cblas.mkl_zero, ptr_z, mkl_m)
                cblas.gemm(Cblas.ColMajor, Cblas.NoTrans, Cblas.NoTrans, \
                    mkl_n, mkl_k, mkl_m, \
                    cblas.mkl_one, ptr_a, mkl_n, ptr_z, mkl_m, \
                    cblas.mkl_zero, ptr_v, mkl_n)
            else:
                z = numpy.dot(u, self.__a.T)
                v[:,:] = numpy.dot(z, self.__a)
