'''
RAL EIGensolver for real symmetric and Hermitian problems.

'''

from raleigh.ndarray_vectors import NDArrayVectors
import numbers
import numpy
import numpy.linalg as nla
import scipy.linalg as sla

def conjugate(a):
    if isinstance(a[0,0], complex):
        return a.conj().T
    else:
        return a.T

class Options:
    def __init__(self):
        self.block_size = 16
        self.min_opA = True
        self.min_opB = True
        self.err_est = 0

class Problem:
    def __init__(self, arg, A, B = None, prod = None):
        self.__A = A
        self.__B = B
        if B is None:
            self.__type = 'std'
        else:
            if prod is None:
                self.__type = 'gen'
            else:
                self.__type = 'pro'
        if isinstance(arg, (numbers.Number, numpy.ndarray)):
            self.__vector = NDArrayVectors(arg)
        else:
            self.__vector = arg
    def A(self):
        return self.__A
    def B(self):
        return self.__B
    def type(self):
        return self.__type[0]
    def vector(self):
        return self.__vector
##    def set_type(self, problem_type):
##        t = problem_type[0]
##        if t == 's' and B is not None:
##            print('WARNING: B will be ignored')
##        elif t != 's' and B is None:
##            print('WARNING: no B defined, standard problem will be solved')
##        if t != 's' and t != 'g' and t != 'p':
##            print('WARNING: unknown problem type, assumed standard')
##        self.__type = t

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
        self.__min_opB = options.min_opB and problem.type() != 's'
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

    def solve(self):

        problem = self.__problem
        problem_type = problem.type()
        std = (problem_type == 's')
        gen = (problem_type == 'g')
        pro = (problem_type == 'p')
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

        print('X:')
        print(X.data())
        if not std:
            B(X, BX)
        if pro:
            A(BX, AX)
            XAX = AX.dot(BX)
        else:
            A(X, AX)
            XAX = AX.dot(X)
        XBX = BX.dot(X)
        print('XAX:')
        print(XAX)
        print('XBX:')
        print(XBX)
        lmd, q = sla.eigh(XAX, XBX, overwrite_a = True, overwrite_b = True)
        print('lmd:')
        print(lmd)
        print('q:')
        print(q)
        X.mult(q, Z)
        Z.copy(X)
        AX.mult(q, Z)
        Z.copy(AX)
        if not std:
            BX.mult(q, Z)
            Z.copy(BX)
        if pro:
            XAX = AX.dot(BX)
        else:
            XAX = AX.dot(X)
        XBX = BX.dot(X)
##        print('XAX:')
##        print(XAX)
##        print('XBX:')
##        print(XBX)
        da = XAX.diagonal()
        db = XBX.diagonal()
##        print('diag(XAX):')
##        print(da)
##        print('diag(XBX):')
##        print(db)
        lmd = da/db
        print('lmd:')
        print(lmd)
        AX.copy(W)
        if gen:
            W.add(BX, -lmd)
        else:
            W.add(X, -lmd)
        print('AX:')
        print(AX.data())
        print('BX:')
        print(BX.data())
        print('residual:')
        print(W.data())
        W.copy(Y)
        if std:
            s = numpy.sqrt(Y.dots(Y))
            Y.scale(s)
            s = numpy.sqrt(Y.dots(Y))
            XBY = Y.dot(X)
            YBY = Y.dot(Y)
        else:
            B(Y, Z)
            s = numpy.sqrt(Z.dots(Y))
            Y.scale(s)
            Z.scale(s)
            s = numpy.sqrt(Z.dots(Y))
            XBY = Z.dot(X)
            YBY = Z.dot(Y)
        print(s)
        YBX = conjugate(XBY)
        GB = numpy.concatenate((XBX, YBX))
        H = numpy.concatenate((XBY, YBY))
        GB = numpy.concatenate((GB, H), axis = 1)
        print('Gram matrix for (X,Y):')
        print(GB)
        if pro:
            A(Z, AYZ)
            XAY = AYZ.dot(BX)
            YAY = AYZ.dot(Z)
        else:
            A(Y, AYZ)
            XAY = AYZ.dot(X)
            YAY = AYZ.dot(Y)
        YAX = conjugate(XAY)
        GA = numpy.concatenate((XAX, YAX))
        H = numpy.concatenate((XAY, YAY))
        GA = numpy.concatenate((GA, H), axis = 1)
        lmd, Q = sla.eigh(GA, GB)
        print(lmd)
##        L = nla.cholesky(G)
##        print('L:')
##        print(L)

