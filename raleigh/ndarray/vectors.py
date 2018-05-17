'''
Implementation of the abstract class Vectors based on numpy.ndarray

'''

import ctypes
import numbers
import numpy

from sys import platform

#print(platform)

try:
    if platform == 'win32':
        mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
    else:
        mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)
    HAVE_MKL = True
    CblasColMajor = 102
    CblasNoTrans = 111
    CblasTrans = 112
    CblasConjTrans = 113
    print('Using %d MKL threads' % mkl.mkl_get_max_threads())

except:
    HAVE_MKL = False

class Vectors:
    def __init__(self, arg, nvec = 0, data_type = None):
        if isinstance(arg, Vectors):
            self.__data = arg.__data.copy()
        elif isinstance(arg, numpy.ndarray):
            self.__data = arg
        elif isinstance(arg, numbers.Number):
            if data_type is None: # use default data type
                self.__data = numpy.zeros((nvec, arg))
            else:
                self.__data = numpy.zeros((nvec, arg), dtype = data_type)
        else:
            raise ValueError \
            ('wrong argument %s in constructor' % repr(type(arg)))
        if HAVE_MKL:
            dt = self.__data.dtype
            if dt == numpy.float32:
                gemm = mkl.cblas_sgemm
                axpy = mkl.cblas_saxpy
                copy = mkl.cblas_scopy
                scal = mkl.cblas_sscal
                norm = mkl.cblas_snrm2
                norm.restype = ctypes.c_float
                inner = mkl.cblas_sdot
                inner.restype = ctypes.c_float
                mkl_one = ctypes.c_float(1.0)
                mkl_zero = ctypes.c_float(0.0)
            elif dt == numpy.float64:
                gemm = mkl.cblas_dgemm
                axpy = mkl.cblas_daxpy
                copy = mkl.cblas_dcopy
                scal = mkl.cblas_dscal
                norm = mkl.cblas_dnrm2
                norm.restype = ctypes.c_double
                inner = mkl.cblas_ddot
                inner.restype = ctypes.c_double
                mkl_one = ctypes.c_double(1.0)
                mkl_zero = ctypes.c_double(0.0)
        m, n = self.__data.shape
        self.__selected = (0, m)
    def dimension(self):
        return self.__data.shape[1]
    def nvec(self):
        return self.__selected[1]
    def selected(self):
        return self.__selected
    def select(self, nv, first = 0):
        assert nv <= self.__data.shape[0] and first >= 0
        self.__selected = (first, nv)
    def select_all(self):
        self.select(self.__data.shape[0])
    def data_type(self):
        return self.__data.dtype
    def is_complex(self):
        return isinstance(self.__data[0,0], complex)
    def clone(self):
        return Vectors(self) #.__data.copy())
    def new_vectors(self, nv = 0):
        m, n = self.__data.shape
        return Vectors(n, nv, self.__data.dtype)
##        data = numpy.ones((nv, n), dtype = self.__data.dtype)
##        return Vectors(data)
    def zero(self):
        f, n = self.__selected;
        self.__data[f : f + n, :] = 0.0
##    def fill(self, array_or_value):
##        f, n = self.__selected;
##        self.__data[f : f + n, :] = array_or_value
    def fill_random(self):
        iv, nv = self.__selected
        m, n = self.__data.shape
        #data = numpy.zeros((nv, n), dtype = self.__data.dtype)
        self.__data[iv : iv + nv,:] = 2*numpy.random.rand(nv, n) - 1
        #return Vectors(data)
    def fill_orthogonal(self, m):
        iv, nv = self.__selected
        k, n = self.__data.shape
        if n < m:
            print('Warning: number of vectors too large, reducing')
            m = n
        #a = numpy.zeros((m, n), dtype = self.__data.dtype)
        a = self.__data[iv : iv + nv, :]
        a[0,0] = 1.0
        i = 1
        while 2*i < m:
            a[i : 2*i, :i] = a[:i, :i]
            a[:i, i : 2*i] = a[:i, :i]
            a[i : 2*i, i : 2*i] = -a[:i, :i]
            i *= 2
        k = i
        j = 2*i
        if j > n:
            for i in range(k, m):
                a[i, i] = 1.0
            return Vectors(a)
        while j <= n:
            a[:k, i : j] = a[:k, :i]
            i, j = j, 2*j
        j = i//2
        a[k : m,   : j] = a[:(m - k), : j]
        a[k : m, j : i] = -a[:(m - k), j : i]
        #return Vectors(a)
    def data(self, i = None):
        f, n = self.__selected
        if i is None:
            return self.__data[f : f + n, :]
        else:
            return self.__data[f + i, :]
    def append(self, other):
        self.__data = numpy.concatenate((self.__data, other.data()))
        self.select_all()

    # BLAS level 1
    def copy(self, other, ind = None):
        i, n = self.__selected
        j, m = other.__selected
        if ind is None:
            other.__data[j : j + n, :] = self.__data[i : i + n, :]
        else:
            other.__data[j : j + len(ind), :] = self.__data[ind, :]
    def scale(self, s):
        f, n = self.__selected;
        for i in range(n):
            if s[i] != 0.0:
                self.__data[i, :] /= s[i]
    def dots(self, other):
        n = self.__selected[1]
        v = numpy.ndarray((n,), dtype = self.__data.dtype)
        for i in range(n):
            if other.is_complex():
                s =  numpy.dot(other.data(i).conj(), self.data(i).T)
            else:
                s = numpy.dot(other.data(i), self.data(i).T)
            v[i] = s
        return v

    # BLAS level 3
    def dot(self, other):
        if other.is_complex():
            return numpy.dot(other.data().conj(), self.data().T)
        else:
            return numpy.dot(other.data(), self.data().T)
    def multiply(self, q, output):
        f, n = output.__selected;
        n = q.shape[1]
        if output.__data[f : f + n, :].flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(q.T, self.data(), out = output.__data[f : f + n, :])
            return
        print('using non-optimized dot')
        output.__data[f : f + n, :] = numpy.dot(q.T, self.data())
    def add(self, other, s, q = None):
        f, n = self.__selected;
        if numpy.isscalar(s):
            if q is None:
                self.__data[f : f + n, :] += s*other.data()
            else:
                self.__data[f : f + n, :] += s*numpy.dot(q.T, other.data())
            return
        for i in range(n):
            self.__data[f + i, :] += s[i]*other.data(i)

