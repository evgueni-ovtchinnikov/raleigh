'''
Implementation of the abstract class Vectors based on numpy.ndarray

'''

import numbers
import numpy

class NDArrayVectors: #(Vectors):
    def __init__(self, arg, arg2 = 1):
        if isinstance(arg, NDArrayVectors):
            self.__data = arg.__data.copy()
        elif isinstance(arg, numpy.ndarray):
            self.__data = arg
        elif isinstance(arg, numbers.Number):
            self.__data = numpy.ndarray((arg, arg2))
        else:
            raise error('wrong argument %s in constructor' % repr(type(arg)))
        n, m = self.__data.shape
        self.__selected = (0, m)
    def dimension(self):
        return self.__data.shape[0]
    def data_type(self):
        return self.__data.dtype
    def new_vectors(self, nv):
        n, m = self.__data.shape
        data = numpy.ndarray((n, nv), dtype = self.__data.dtype)
        return NDArrayVectors(data)
    def new_orthogonal_vectors(self, m):
        n, k = self.__data.shape
        if n < m:
            print('Warning: number of vectors too large, reducing')
            m = n
        a = numpy.zeros((n, m))
        a[0,0] = 1.0
        i = 1
        while 2*i < m:
            a[i : 2*i, :i] = a[:i, :i]
            a[:i, i : 2*i] = a[:i, :i]
            a[i : 2*i, i : 2*i] = -a[:i, :i]
            i *= 2
        k = i
        j = 2*i
        while j <= n:
            a[i : j, :i] = a[:i, :i]
            i, j = j, 2*j
        j = i/2
        a[ : j, k : m] = a[ : j, :(m - k)]
        a[j : i, k : m] = -a[j : i, :(m - k)]
        return NDArrayVectors(a)
    def selected(self):
        return self.__selected
    def select(self, first, nv):
        self.__selected = (first, nv)
    def fill(self, array):
        f, n = self.__selected;
        self.__data[:, f : f + n] = array
    def data(self, i = None):
        f, n = self.__selected
        if i is None:
            return self.__data[:, f : f + n]
        else:
            return self.__data[:, f + i]
    def copy(self, other):
        i, n = self.__selected
        j, m = other.__selected
        other.__data[:, j : j + n] = self.__data[:, i : i + n]
    def dot(self, other):
        if isinstance(other.data(), complex):
            return numpy.dot(other.data().conj().T, self.data())
        else:
            return numpy.dot(other.data().T, self.data())
    def dots(self, other):
        n = self.__selected[1]
        v = numpy.ndarray((n,), dtype = self.__data.dtype)
        for i in range(n):
            if isinstance(other.data(i), complex):
                s =  numpy.dot(other.data(i).conj().T, self.data(i))
            else:
                s = numpy.dot(other.data(i).T, self.data(i))
            v[i] = s
        return v
    def mult(self, q, output):
        f, n = output.__selected;
        output.__data[:, f : f + n] = numpy.dot(self.data(), q)
    def scale(self, s):
        f, n = self.__selected;
        for i in range(n):
            if s[i] != 0.0:
                self.__data[:, i] /= s[i]
    def add(self, other, s):
        f, n = self.__selected;
        for i in range(n):
            self.__data[:, i] += s[i]*other.data(i)

