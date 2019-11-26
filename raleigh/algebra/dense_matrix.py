# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Architecture-aware wrap for a dense matrix.
"""

import numpy


class AMatrix:

    def __init__(self, a, arch='cpu', copy_data=False):
        self.__arch = arch
        if arch[:3] == 'gpu':
            try:
                from . import cuda_wrap as cuda
                from .dense_cublas import Matrix, Vectors
                self.__op = Matrix(a)
                self.__gpu = cuda
            except:
                if len(arch) > 3 and arch[3] == '!':
                    raise RuntimeError('cannot use GPU')
        else:
            from .dense_cpu import Matrix, Vectors
#            from .dense_numpy import Matrix, Vectors
            if copy_data:
                self.__op = Matrix(a.copy())
            else:
                self.__op = Matrix(a)
            self.__gpu = None
        self.__Vectors = Vectors
        self.__vectors = None
        vmin = numpy.amin(a)
        vmax = numpy.amax(a)
        self.__scale = max(abs(vmin), abs(vmax))

    def as_operator(self):
        return self.__op

    def as_vectors(self):
        if self.__vectors is None:
            self.__vectors = self.__Vectors(self.__op, shallow=True)
        return self.__vectors

    def arch(self):
        return self.__arch

    def gpu(self):
        return self.__gpu

    def dots(self):
        return self.__op.dots()

    def data_type(self):
        return self.__op.data_type()

    def shape(self):
        return self.__op.shape()

    def scale(self):
        return self.__scale
