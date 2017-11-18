'''
Implementation of the abstract class Vectors based on numpy.ndarray

'''

import numbers
import numpy

ORDER = 'C'

class NDArrayVectors: #(Vectors):
    def __init__(self, arg, arg2 = 1):
        if isinstance(arg, NDArrayVectors):
            self.__data = arg.__data.copy()
        elif isinstance(arg, numpy.ndarray):
            self.__data = arg
        elif isinstance(arg, numbers.Number):
            self.__data = numpy.zeros((arg2, arg), order = ORDER)
        else:
            raise ValueError \
            ('wrong argument %s in constructor' % repr(type(arg)))
        m, n = self.__data.shape
        self.__selected = (0, m)
    def dimension(self):
        return self.__data.shape[1]
    def data_type(self):
        return self.__data.dtype
    def is_complex(self):
        return isinstance(self.__data[0,0], complex)
    def new_vectors(self, nv):
        m, n = self.__data.shape
        data = numpy.zeros((nv, n), dtype = self.__data.dtype, order = ORDER)
        return NDArrayVectors(data)
    def new_orthogonal_vectors(self, m):
        k, n = self.__data.shape
        if n < m:
            print('Warning: number of vectors too large, reducing')
            m = n
        a = numpy.zeros((m, n), order = ORDER)
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
            return NDArrayVectors(a)
        while j <= n:
            a[:i, i : j] = a[:i, :i]
            i, j = j, 2*j
        j = i//2
        a[k : m,   : j] = a[:(m - k), : j]
        a[k : m, j : i] = -a[:(m - k), j : i]
        return NDArrayVectors(a)
    def nvec(self):
        return self.__selected[1]
    def selected(self):
        return self.__selected
    def select(self, first, nv):
        self.__selected = (first, nv)
    def fill(self, array):
        f, n = self.__selected;
        self.__data[f : f + n, :] = array
    def data(self, i = None):
        f, n = self.__selected
        if i is None:
            return self.__data[f : f + n, :]
        else:
            return self.__data[f + i, :]
    def copy(self, other, ind = None):
        i, n = self.__selected
        j, m = other.__selected
        if ind is None:
            other.__data[j : j + n, :] = self.__data[i : i + n, :]
        else:
            other.__data[j : j + len(ind), :] = self.__data[ind, :]
    def dot(self, other):
        if other.is_complex():
            return numpy.dot(other.data().conj(), self.data().T)
        else:
            return numpy.dot(other.data(), self.data().T)
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
    def mult(self, q, output):
        f, n = output.__selected;
        if output.__data[f : f + n, :].flags['C_CONTIGUOUS']:
            #print('using optimized dot')
            numpy.dot(q.T, self.data(), out = output.__data[f : f + n, :])
            return
        print('using non-optimized dot')
        output.__data[f : f + n, :] = numpy.dot(q.T, self.data())
#    def axpy(self, q, output):
#        f, n = output.__selected;
#        output.__data[:, f : f + n] += numpy.dot(self.data(), q)
    def scale(self, s):
        f, n = self.__selected;
        for i in range(n):
            if s[i] != 0.0:
                self.__data[i, :] /= s[i]
    def add(self, other, s):
        if numpy.isscalar(s):
            self.__data += s*other.data()
            return
        f, n = self.__selected;
        for i in range(n):
            self.__data[i, :] += s[i]*other.data(i)

