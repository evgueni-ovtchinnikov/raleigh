'''
RAL EIGensolver for real symmetric and Hermitian problems.

'''

import ndarray_vectors
import numpy
import scipy.linalg as la

class Options:
    def __init__(self):
        self.block_size = 16
        self.min_opA = True
        self.min_opB = True
        self.err_est = 0

class Problem:
    def __init__(self, vector, A, B = None):
        self.__A = A
        self.__B = B
        if B is None:
            self.__type = 'std'
        else:
            self.__type = 'gen'
        if isinstance(vector, numpy.ndarray):
            self.__vector = NDArrayVectors(vector)
        else:
            self.__vector = vector
    def A():
        return self.__A
    def B():
        return self.__B
    def type():
        return self.__type
    def vector():
        return self.__vector
    def set_type(self, problem_type):
        t = problem_type[0]
        if t == 's' and B is not None:
            print('WARNING: B will be ignored')
        elif t != 's' and B is None:
            print('WARNING: no B defined, standard problem will be solved')
        if t != 's' and t != 'g' and t != 'p':
            print('WARNING: unknown problem type, assumed standard')
        self.__type = t

class Solver:
    def __init__(self, problem, options = Options()):
        vector = problem.vector()
        self.__problem = problem
        self.__data_type = vector.data_type()
        m = int(options.block_size)
        mm = m + m
        self.__block_size = m
        self.lmd = numpy.ndarray((m,), dtype = numpy.float64)
        self.res = numpy.ndarray((m,), dtype = numpy.float64)
        self.__ind = numpy.ndarray((m,), dtype = numpy.int32)
        self.__lmd = numpy.ndarray((mm,), dtype = numpy.float64)
        self.__min_opA = options.min_opA
        self.__min_opB = options.min_opB and problem != 's'
        self.__err_est = options.err_est
        self.__X = vector.new_orthogonal_vectors(m)
        self.__Y = vector.new_vectors(m)
        self.__Z = vector.new_vectors(m)
        self.__W = vector.new_vectors(m)
        blocks = 4
        if self.__min_opA:
            self.__AX = vector.new_vectors(m)
            self.__AYZ = vector.new_vectors(m)
            blocks += 2
        else:
            self.__AX = self.__W
            self.__AYZ = self.__W
        if self.__min_opB or problem.type() == 'p':
            self.__BX = vector.new_vectors(m)
            self.__BYZ = vector.new_vectors(m)
            blocks += 2
        else:
            if problem.type() == 's':
                self.__BX = self.__X
            else:
                self.__BX = self.__Y
            self.__BYZ = self.__Y                
        self.__blocks = blocks
        self.__P = None

    def set_preconditioner(self, P):
        self.__P = P

    def solve():

        problem = self.__problem
        X = self.__X
        Y = self.__Y
        Z = self.__Z
        W = self.__W
        AX = self.__AX
        BX = self.__BX
        AYZ = self.__AYZ
        BYZ = self.__BYZ
        A = self.__problem.A()
        B = self.__problem.B()
        P = self.__P

        A(X, AX)
        if problem.type() != 's':
            B(X, BX)
        XAX = AX.dot(X)
        XBX = BX.dot(X)
        print('XAX:')
        print(XAX)
        print('XBX:')
        print(XBX)
