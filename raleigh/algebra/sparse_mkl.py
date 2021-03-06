# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Wrapper for MKL sparse symmetric/Hermitian matrices and solvers working with
   SciPy sparse matrices.
"""

import numpy
import scipy.sparse as scs

from .mkl_wrap import SparseSymmetricMatrix as SSM
from .mkl_wrap import ParDiSo as SSS
from .mkl_wrap import ILUT


class SparseSymmetricMatrix:

    def __init__(self, matrix):
        try:
            csr = matrix.csr()
        except:
            csr = scs.triu(matrix, format='csr')
            csr.sort_indices()
        a = csr.data
        ia = csr.indptr + 1
        ja = csr.indices + 1
        self.__csr = csr
        self.__ssm = SSM(a, ia, ja)
        self.__a = a
        self.__ia = ia
        self.__ja = ja

    def size(self):
        return self.__csr.shape[0]

    def data_type(self):
        return self.__a.dtype

    def csr(self):
        return self.__csr

    def apply(self, x, y):
        try:
            x = x.data()
            y = y.data()
        except:
            pass
        self.__ssm.dot(x, y)


class SparseSymmetricSolver:

    def __init__(self, dtype=numpy.float64, pos_def=False):
        self.__solver = SSS(dtype=dtype, pos_def=pos_def)
        self.__dtype = dtype

    def analyse(self, a, sigma=0, b=None):
        data = a.data
        if sigma != 0:
            if b is None:
                b = scs.eye(a.shape[0], dtype=a.data.dtype, format='csr')
            a_s = a - sigma * b
        else:
            a_s = a
        a_s = scs.triu(a_s, format='csr')
        a_s.sort_indices()
        ia = a_s.indptr + 1
        ja = a_s.indices + 1
        data = a_s.data
        try:
            status = self.__solver.analyse(data, ia, ja)
            if status < 0:
                msg = 'sparse factorization returned error %d' % status
                raise RuntimeError(msg)
            self.__n = ia.shape[0] - 1
            self.__sigma = sigma
        except:
            raise RuntimeError('factorization failed on analysis stage')

    def factorize(self):
        try:
            status = self.__solver.factorize()
            if status < 0:
                msg = 'sparse factorization returned error %d' % status
                raise RuntimeError(msg)
        except:
            raise RuntimeError('factorization failed (near singular matrix?)')

    def solve(self, b, x):
        try:
            b = b.data()
            x = x.data()
        except:
            pass
        try:
            status = self.__solver.solve(b, x)
            if status < 0:
                msg = 'sparse solver returned error %d' % status
                raise RuntimeError(msg)
        except:
            raise RuntimeError('solution failed (near singular matrix?)')

    def apply(self, b, x):
        self.solve(b, x)

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


class IncompleteLU:

    def __init__(self, matrix):
        matrix = matrix.tocsr().sorted_indices()
        a = matrix.data
        ia = matrix.indptr + 1
        ja = matrix.indices + 1
        self.__ilut = ILUT(a, ia, ja)

    def factorize(self, tol=1e-6, max_fill=1):
        self.__ilut.factorize(tol=tol, max_fill_rel=max_fill)

    def apply(self, x, y):
        try:
            x = x.data()
            y = y.data()
        except:
            pass
        self.__ilut.solve(x, y)


class Operator:

    def __init__(self, op):
        self.__op = op

    def apply(self, x, y):
        try:
            x = x.data()
            y = y.data()
        except:
            pass
        self.__op.apply(x, y)
