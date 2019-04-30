"""
MKL implementation of sparse symmetric/Hermitian matrices and solvers.

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import numpy
import scipy.sparse as scs

from .mkl import SparseSymmetricMatrix as SSM
from .mkl import ParDiSo as SSS


class SparseSymmetricMatrix:
    def __init__(self, matrix):
        csr = scs.triu(matrix, format = 'csr')
        csr.sort_indices()
        a = csr.data
        ia = csr.indptr + 1
        ja = csr.indices + 1
        self.__csr = csr
        self.__ssm = SSM(a, ia, ja)
    def apply(self, x, y):
        self.__ssm.apply(x.data(), y.data())


class SparseSymmetricSolver:
    def __init__(self, dtype = numpy.float64, pos_def = False):
        self.__solver = SSS(dtype = dtype, pos_def = pos_def)
        self.__dtype = dtype
    def analyse(self, a, sigma = 0, b = None):
        data = a.data
        if sigma != 0:
            if b is None:
                b = scs.eye(a.shape[0], dtype = a.data.dtype, format = 'csr')
            a_s = a - sigma * b
        else:
            a_s = a
        a_s.sort_indices()
        ia = a_s.indptr + 1
        ja = a_s.indices + 1
        data = a_s.data
        self.__solver.analyse(data, ia, ja)
        self.__n = ia.shape[0] - 1
        self.__sigma = sigma
    def factorize(self):
        self.__solver.factorize()
    def solve(self, b, x):
        self.__solver.solve(b.data(), x.data())
    def inertia(self):
        return self.__solver.inertia()
    def size(self):
        return self.__n
    def data_type(self):
        return self.__dtype
    def sigma(self):
        return self.__sigma
    def solver(self):
        return self.__solver
