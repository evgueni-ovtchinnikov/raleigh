'''
RAL EIGensolver for real symmetric and Hermitian problems.

'''

from raleigh.piv_chol import piv_chol
import numpy
import numpy.linalg as nla
import scipy.linalg as sla

def conjugate(a):
    if isinstance(a[0,0], complex):
        return a.conj().T
    else:
        return a.T

def transform(A, U):
    B = sla.solve_triangular(conjugate(U), conjugate(A), lower = True)
    A = sla.solve_triangular(conjugate(U), conjugate(B), lower = True)
    return A

class Options:
    def __init__(self):
        self.block_size = 16
#        self.min_opA = True
#        self.min_opB = True
        self.err_est = 0
        self.max_iter = 10

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
            raise ValueError('wrong number of needed eigenpairs')

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
#        self.__min_opA = options.min_opA
#        self.__min_opB = options.min_opB and problem.type() != 's'
        self.__err_est = options.err_est
        self.__max_iter = options.max_iter
        self.__X = vector.new_orthogonal_vectors(m)
        self.__Y = vector.new_vectors(m)
        self.__Z = vector.new_vectors(m)
        self.__W = vector.new_vectors(m)
        self.__AX = vector.new_vectors(m)
        self.__AY = vector.new_vectors(m)
        if problem.type() != 's':
            self.__BX = vector.new_vectors(m)
            self.__BY = vector.new_vectors(m)
        else:
            self.__BX = self.__X
            self.__BY = self.__Y
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
#        minA = self.__min_opA
#        minB = self.__min_opB
#        m = self.__block_size
        X = self.__X
        Y = self.__Y
        Z = self.__Z
        W = self.__W
        AX = self.__AX
        BX = self.__BX
        AY = self.__AY
        BY = self.__BY
        AZ = None
        BZ = None
        leftX = self.__leftX
        rightX = self.__rightX

#        print('X:')
#        print(X.data())
        if not std:
            B(X, BX)
        else:
            BX = X
        if pro:
            A(BX, AX)
            XAX = AX.dot(BX)
        else:
            A(X, AX)
            XAX = AX.dot(X)
        XBX = BX.dot(X)
        lmd, q = sla.eigh(XAX, XBX, overwrite_a = True, overwrite_b = True)
        print('lmd:')
        print(lmd)
        X.mult(q, W)
        W.copy(X)
        AX.mult(q, W)
        W.copy(AX)
        if not std:
            BX.mult(q, Z)
            Z.copy(BX)

## TODO: the main loop starts here
        for iteration in range(self.__max_iter):
            print('------------- iteration %d' % iteration)
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
            AX.copy(W)
            if gen:
                W.add(BX, -lmd)
            else:
                W.add(X, -lmd)
            s = W.dots(W)
            print('residual norms:')
            print(numpy.sqrt(s))
    
    ## TODO: estimate errors, 
    ##       move converged X to Xc, 
    ##       select Xs, AXs, BXs and Ws accordingly
            nx = X.nvec()
            W.copy(Y)
            if std:
                s = numpy.sqrt(Y.dots(Y))
                Y.scale(s)
                s = numpy.sqrt(Y.dots(Y))
                XBY = Y.dot(X)
                YBY = Y.dot(Y)
            else:
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
    #        print('Gram matrix for (X,Y):')
    #        print(GB)
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
            Y.copy(W, indy)
            W.copy(Y)
            Y.select(ny)
            AY.select(ny)
            if not std:
                BY.copy(W, indy)
                W.copy(BY)
                BY.select(ny)
#                XBY = BY.dot(X)
#                YBY = BY.dot(Y)
#            else:
#                XBY = Y.dot(X)
#                YBY = Y.dot(Y)
#            YBX = conjugate(XBY)
#            GB = numpy.concatenate((XBX, YBX))
#            H = numpy.concatenate((XBY, YBY))
#            GB = numpy.concatenate((GB, H), axis = 1)
#            err = numpy.dot(U.T, U) - GB
#            print('UTU err:', nla.norm(err))
    
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
            GA = transform(GA, U)
            lmd, Q = sla.eigh(GA)
            print(lmd)
            Q = sla.solve_triangular(U, Q)
            QX = numpy.concatenate((Q[:, :leftX], Q[:, nxy - rightX:]), axis = 1)
            QZ = Q[:, leftX : nxy - rightX]
#            Q = numpy.concatenate((QX, QZ), axis = 1)
            QXX = QX[:nx, :].copy()
            QYX = QX[nx:, :].copy()
            QXZ = QZ[:nx, :].copy()
            QYZ = QZ[nx:, :].copy()
    #        print(QXX.flags['C_CONTIGUOUS'])
    #        print(QYX.flags['C_CONTIGUOUS'])
    #        print(QXZ.flags['C_CONTIGUOUS'])
    #        print(QYZ.flags['C_CONTIGUOUS'])
#            QX = numpy.concatenate((QXX, QYX))
#            G = numpy.dot(QX.T, numpy.dot(GB, QX))
#            print(G.diagonal())
#            #I = numpy.dot(QX.T, QX)
#            I = BX.dot(X)
#            print(I.diagonal())
    
    #        nx_new = leftX + rightX
    #        nz = nxy - nx_new
    #        if minA:
            AX.mult(QXX, W)
            AY.mult(QYX, Z)
            W.add(Z, 1.0)
            AY.mult(QYZ, Z)
            AZ = AY
            AX.mult(QXZ, AZ)
            W.copy(AX)
            AZ.add(Z, 1.0)
    #        if minB:
            if not std:
                BX.mult(QXX, W)
                BY.mult(QYX, Z)
                W.add(Z, 1.0)
                BY.mult(QYZ, Z)
                BZ = BY
                BX.mult(QXZ, BZ)
                W.copy(BX)
                BZ.add(Z, 1.0)
            else:
                BZ = Z
            X.mult(QXX, W) # W = X*QXX
            Y.mult(QYX, Z)
            W.add(Z, 1.0)
            X.mult(QXZ, Z) # Z = X*QXZ
            W.copy(X)      # X = W
            Y.mult(QYZ, W) # Z += Y*QYZ
            Z.add(W, 1.0)

#        A(X, AX)
        if pro:
            XAX = AX.dot(BX)
        else:
            XAX = AX.dot(X)
        XBX = BX.dot(X)
        da = XAX.diagonal()
        db = XBX.diagonal()
#        print('diag(XAX):')
#        print(da)
#        print('diag(XBX):')
#        print(db)
        lmd = da/db
        print('lmd:')
        print(lmd)
        AX.copy(W)
        if gen:
            W.add(BX, -lmd)
        else:
            W.add(X, -lmd)
        s = W.dots(W)
        print('residual norms:')
        print(numpy.sqrt(s))

#        if pro:
#            ZAZ = AZ.dot(BZ)
#        else:
#            ZAZ = AZ.dot(Z)
#        ZBZ = BZ.dot(Z)
#        da = ZAZ.diagonal()
#        db = ZBZ.diagonal()
#        print('diag(ZAZ):')
#        print(da)
#        print('diag(ZBZ):')
#        print(db)
#        lmd = da/db
#        print('lmd:')
#        print(lmd)
