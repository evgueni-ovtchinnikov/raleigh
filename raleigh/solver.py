'''
RAL EIGensolver for real symmetric and Hermitian problems.

'''

from raleigh.ndarray_vectors import NDArrayVectors
from raleigh.piv_chol import piv_chol
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
    def __init__(self, v, A, B = None, prod = None):
        self.__vector = v
        self.__A = A
        self.__B = B
        if B is None:
            self.__type = 'std'
        else:
            if prod is None:
                self.__type = 'gen'
            else:
                self.__type = 'pro'
#        if isinstance(arg, (numbers.Number, numpy.ndarray)):
#            self.__vector = NDArrayVectors(arg)
#        else:
#            self.__vector = arg
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
    def __init__(self, problem, options = Options(), which = (-1,-1)):
        vector = problem.vector()
        self.__problem = problem
        self.__data_type = vector.data_type()

        left = int(which[0])
        right = int(which[1])
        if left == 0 and right == 0:
            raise ValueError('wrong nuber of needed eigenpairs')

        m = int(options.block_size)
        if m < 0:
            if left < 0 or right < 0:
                m = 16
            else:
                m = left + right
        if m < 2:
            raise ValueError('block size must be at least 2')
        mm = m + m
        self.__block_size = m

        if left == 0:
            self.__leftX = 0
        elif right == 0:
            self.__leftX = m
        elif left > 0 and right > 0:
            q = m*(left/(left + 1.0*right))
            self.__leftX = max(1, int(round(q)))
        else:
            self.__leftX = m//2
        self.__rightX = m - self.__leftX
        print(self.__leftX, self.__rightX)
            
        self.lmd = numpy.ndarray((m,), dtype = numpy.float64)
        self.res = numpy.ndarray((m,), dtype = numpy.float64)
        self.__ind = numpy.ndarray((m,), dtype = numpy.int32)
        self.__lmd = numpy.ndarray((mm,), dtype = numpy.float64)
        self.__min_opA = options.min_opA
        self.__min_opB = options.min_opB and problem.type() != 's'
        self.__err_est = options.err_est
        self.__X = vector.new_orthogonal_vectors(m)
#        self.__Y = vector.new_vectors(m)
#        self.__Z = vector.new_vectors(m)
#        self.__W = vector.new_vectors(m)
        blocks = 4
        if self.__min_opA:
            self.__AX = vector.new_vectors(m)
            #self.__AYZ = vector.new_vectors(m)
            blocks += 2
        else:
            self.__AX = None
#            self.__AX = self.__W
#            #self.__AYZ = self.__W
        if self.__min_opB or problem.type() == 'p':
            self.__BX = vector.new_vectors(m)
            #self.__BYZ = vector.new_vectors(m)
            blocks += 2
        else:
            self.__BX = None
#            if problem.type() == 's':
#                self.__BX = self.__X
#            else:
#                self.__BX = self.__Y
#            #self.__BYZ = self.__Z                
        self.__blocks = blocks
        self.__P = None

    def set_preconditioner(self, P):
        self.__P = P

    def solve(self):

        problem = self.__problem
        problem_type = problem.type()
        vector = problem.vector()
        std = (problem_type == 's')
        gen = (problem_type == 'g')
        pro = (problem_type == 'p')
        A = self.__problem.A()
        B = self.__problem.B()
        P = self.__P
        minA = self.__min_opA
        minB = self.__min_opB
        m = self.__block_size
        X = self.__X
        #Y = self.__Y
        #Z = self.__Z
        #W = self.__W
        AX = self.__AX
        BX = self.__BX
        Z = None
        AZ = None
        BZ = None
        #AYZ = self.__AYZ
        #BYZ = self.__BYZ
        leftX = self.__leftX
        rightX = self.__rightX

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
#        print('XAX:')
#        print(XAX)
#        print('XBX:')
#        print(XBX)
        lmd, q = sla.eigh(XAX, XBX, overwrite_a = True, overwrite_b = True)
        print('lmd:')
        print(lmd)
#        print('q:')
#        print(q)
        W = vector.new_vectors(m)
        X.mult(q, W)
        W.copy(X)
        AX.mult(q, W)
        W.copy(AX)
        if not std:
            BX.mult(q, W)
            W.copy(BX)
        del W

## TODO: the main loop starts here
        if pro:
            XAX = AX.dot(BX)
        else:
            XAX = AX.dot(X)
        XBX = BX.dot(X)
        da = XAX.diagonal()
        db = XBX.diagonal()
        lmd = da/db
        print('lmd:')
        print(lmd)
        nx = AX.selected()[1]
        W = vector.new_vectors(nx)
        AX.copy(W)
        if gen:
            W.add(BX, -lmd)
        else:
            W.add(X, -lmd)

## TODO: estimate errors, 
##       move converged X to Xc, 
##       select Xs, AXs, BXs and Ws accordingly
        nx = X.nvec()
        Y = vector.new_vectors(nx)
        W.copy(Y)
        if std:
            s = numpy.sqrt(Y.dots(Y))
            Y.scale(s)
            s = numpy.sqrt(Y.dots(Y))
            XBY = Y.dot(X)
            YBY = Y.dot(Y)
        else:
            BY = vector.new_vectors(nx)
            B(Y, BY)
            s = numpy.sqrt(BY.dots(Y))
            Y.scale(s)
            BY.scale(s)
            s = numpy.sqrt(BY.dots(Y))
            XBY = BY.dot(X)
            YBY = BY.dot(Y)
#        print(s)
        YBX = conjugate(XBY)
        GB = numpy.concatenate((XBX, YBX))
        H = numpy.concatenate((XBY, YBY))
        GB = numpy.concatenate((GB, H), axis = 1)
        print('Gram matrix for (X,Y):')
        print(GB)
        U = GB
        ny = Y.nvec() # = nx
        ind, dropped, last_piv = piv_chol(U, nx, 1e-4)
        print(ind)
        print(dropped)
        ny -= dropped
        nxy = nx + ny
        U = U[:nxy, :nxy]
        indy = ind[nx: nxy]
        for i in range(ny):
            indy[i] -= nx
        if W.nvec() != ny:
            del W
            W = vector.new_vectors(ny)
        Y.copy(W, indy)
        if Y.nvec() != ny:
            del Y
            Y = vector.new_vectors(ny)
        W.copy(Y)
        if not std:
            BY.copy(W, indy)
            if BY.nvec() != ny:
                del BY
                BY = vector.new_vectors(ny)
            W.copy(BY)

        AY = vector.new_vectors(ny)
        if pro:
            A(BY, AY)
            XAY = AY.dot(BX)
            YAY = AY.dot(BY)
        else:
            A(Y, AY)
            XAY = AY.dot(X)
            YAY = AY.dot(Y)
        YAX = conjugate(XAY)
        GA = numpy.concatenate((XAX, YAX))
        H = numpy.concatenate((XAY, YAY))
        GA = numpy.concatenate((GA, H), axis = 1)
        GA = numpy.dot(U.T, numpy.dot(GA, U))
        lmd, Q = sla.eigh(GA)
        print(lmd)
        QX = numpy.concatenate((Q[:, :leftX], Q[:, nxy - rightX:]), axis = 1)
        QZ = Q[:, leftX : nxy - rightX]
        QXX = QX[:nx, :].copy()
        QYX = QX[nx:, :].copy()
        QXZ = QZ[:nx, :].copy()
        QYZ = QZ[nx:, :].copy()
        print(QXX.flags['C_CONTIGUOUS'])
        print(QYX.flags['C_CONTIGUOUS'])
        print(QXZ.flags['C_CONTIGUOUS'])
        print(QYZ.flags['C_CONTIGUOUS'])

        nx_new = leftX + rightX
        nz = nxy - nx_new
        if W.nvec() != nx_new:
            del W
            W = vector.new_vectors(nx_new)
        #print(W.data().flags['C_CONTIGUOUS'])
        Z = vector.new_vectors(nx_new)
        if minA:
            AX.mult(QXX, W)
            AY.mult(QYX, Z)
            W.add(Z, 1.0)
            if Z.nvec() != nz:
                del Z
                Z = vector.new_vectors(nz)
            AY.mult(QYZ, Z)
            del AY
            AZ = vector.new_vectors(nz)
            AX.mult(QXZ, AZ)
            W.copy(AX)
            AZ.add(Z, 1.0)
        if minB:
            BX.mult(QXX, W)
            if Z.nvec() != nx_new:
                del Z
                Z = vector.new_vectors(nx_new)
            BY.mult(QYX, Z)
            W.add(Z, 1.0)
            if Z.nvec() != nz:
                del Z
                Z = vector.new_vectors(nz)
            BY.mult(QYZ, Z)
            del BY
            BZ = vector.new_vectors(nz)
            BX.mult(QXZ, BZ)
            W.copy(BX)
            BZ.add(Z, 1.0)
        X.mult(QXX, W) # W = X*QXX
        Y.axpy(QYX, W) # W += Y*QYX
        if Z.nvec() != nz:
            del Z
            Z = vector.new_vectors(nz)
        X.mult(QXZ, Z) # Z = X*QXZ
        Y.axpy(QYZ, Z) # Z += Y*QYZ
        W.copy(X)      # X = W

        if pro:
            XAX = AX.dot(BX)
        else:
            XAX = AX.dot(X)
        XBX = BX.dot(X)
        da = XAX.diagonal()
        db = XBX.diagonal()
        print('diag(XAX):')
        print(da)
        print('diag(XBX):')
        print(db)
        lmd = da/db
        print('lmd:')
        print(lmd)
