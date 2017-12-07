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
        self.res_tol = 1e-2
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
    def __init__ \
    (self, problem, eigenvectors, options = Options(), which = (-1,-1)):
        vector = problem.vector()
        self.__problem = problem
        problem_type = problem.type()
        std = (problem_type == 's')
        self.__data_type = vector.data_type()

        left = int(which[0])
        right = int(which[1])
        if left == 0 and right == 0:
            raise ValueError('wrong number of needed eigenpairs')
        self.__left = left
        self.__right = right

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

        self.eigenvalues = numpy.ndarray((0,), dtype = numpy.float64)
        self.lmd = numpy.ndarray((m,), dtype = numpy.float64)
        self.res = numpy.ndarray((m,), dtype = numpy.float64)
        self.__err_est = options.err_est
        self.__res_tol = options.res_tol
        self.__max_iter = options.max_iter
        self.__lcon = 0
        self.__rcon = 0
        self.__Xc = eigenvectors
        if not std:
            self.__BXc = eigenvectors.clone()
        self.__ind = numpy.ndarray((m,), dtype = numpy.int32)
        self.__lmd = numpy.ndarray((mm,), dtype = numpy.float64)
#        self.__min_opA = options.min_opA
#        self.__min_opB = options.min_opB and problem.type() != 's'
        self.__X = vector.new_orthogonal_vectors(m)
        self.__Y = vector.new_vectors(m)
        self.__Z = vector.new_vectors(m)
        self.__W = vector.new_vectors(m)
        self.__AX = vector.new_vectors(m)
        self.__AY = vector.new_vectors(m)
        if problem.type() != 's':
            self.__BXc = eigenvectors.new_vectors(0)
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
        #vector = problem.vector()
        std = (problem_type == 's')
        gen = (problem_type == 'g')
        pro = (problem_type == 'p')
        A = self.__problem.A()
        B = self.__problem.B()
        P = self.__P
#        minA = self.__min_opA
#        minB = self.__min_opB
        m = self.__block_size
        left = self.__left
        right = self.__right
        Xc = self.__Xc
        if not std:
            BXc = self.__BXc
        lmd = self.lmd
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
        ix = 0
        nx = self.__block_size
        nz = 0
        res_tol = self.__res_tol

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
        lmd, Q = sla.eigh(XAX, XBX, overwrite_a = True, overwrite_b = True)
        print('lmd:')
        print(lmd)
        X.mult(Q, W)
        W.copy(X)
        AX.mult(Q, W)
        W.copy(AX)
        if not std:
            BX.mult(Q, Z)
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
            
            # compute residuals
            # std: A X - X lmd
            # gen: A X - B X lmd
            # pro: A B X - X lmd
            W.select(nx, ix)
            Y.select(nx)
            AX.copy(W)
            if gen:
                W.add(BX, -lmd)
            else:
                W.add(X, -lmd)

            if Xc.nvec() > 0:
                # orthogonalize W to Xc
                # std: W := W - Xc Xc* W
                # gen: W := W - BXc Xc* W
                # pro: W := W - Xc BXc* W
                if pro:
                    Q = W.dot(BXc)
                else:
                    Q = W.dot(Xc)
                if gen:
                    BXc.mult(Q, Y)
                else:
                    Xc.mult(Q, Y)
                W.add(Y, -1.0)
            s = W.dots(W)
            self.res = numpy.sqrt(s)
            print('residual norms:')
            print(self.res)
    
            ## TODO: estimate errors, 
            lcon = 0
            for i in range(leftX):
                j = self.__lcon + ix + i
                if self.res[i] < res_tol:
                    print('left eigenvector %d converged' % j)
                    lcon += 1
                else:
                    break
            self.__lcon += lcon
            rcon = 0
            for i in range(rightX):
                j = self.__rcon + m - ix - nx + i
                if self.res[ix + nx - i - 1] < res_tol:
                    print('right eigenvector %d converged' % j)
                    rcon += 1
                else:
                    break
            self.__rcon += rcon
            ##       move converged X to Xc,
            if lcon > 0:
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[ix : ix + lcon]))
                #print(self.eigenvalues)
                X.select(lcon, ix)
                Xc.append(X)
                if not std:
                    BX.select(lcon, ix)
                    BXc.append(BX)
            if rcon > 0:
                jx = ix + nx
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[jx - rcon : jx]))
                X.select(rcon, jx - rcon)
                Xc.append(X)
                if not std:
                    BX.select(rcon, jx - rcon)
                    BXc.append(BX)
            if self.__lcon >= left and self.__rcon >= right:
                break
            #print(Xc.dimension(), Xc.nvec())
            #print(BXc.dimension(), BXc.nvec())
            ##       select Xs, AXs, BXs accordingly
            #nx = X.nvec()
            ix += lcon
            nx -= lcon + rcon
            X.select(nx, ix)
            AX.select(nx, ix)
            if not std:
                BX.select(nx, ix)
            XAX = XAX[ix : ix + nx, ix : ix + nx]
            XBX = XBX[ix : ix + nx, ix : ix + nx]

            W.copy(Y)
            
            if nz > 0:
                # compute the conjugation matrix
                if pro:
                    W.select(Y.nvec())
                    B(Y, W)
                    ZAY = W.dot(AZ)
                else:
                    ZAY = Y.dot(AZ)
                if std:
                    ZBY = Y.dot(Z)
                else:
                    ZBY = Y.dot(BZ)
                Num = ZAY - numpy.dot(ZBY, numpy.diag(lmd))
                ny = Y.nvec()
                Lmd = numpy.ndarray((1, ny))
                Mu = numpy.ndarray((nz, 1))
                Lmd[0, :] = lmd
                Mu[:, 0] = lmdz
                Den = Mu - Lmd
                Beta = Num/Den
                
                # conjugate search directions
                AZ.select(ny)
                Z.mult(Beta, AZ)
                Y.add(AZ, -1.0)
                if pro:
                    BZ.mult(Beta, AZ)
                    W.add(AZ, -1.0)
                    BY.select(ny)
                    W.copy(BY)

            if Xc.nvec() > 0 and (P is not None or gen):
                # orthogonalize Y to Xc
                # std: W := W - Xc Xc* W (not needed if P is None)
                # gen: W := W - Xc BXc* W
                # pro: W := W - Xc BXc* W (not needed if P is None)
                if not std:
                    Q = Y.dot(BXc)
                else:
                    Q = Y.dot(Xc)
                Xc.mult(Q, W)
                Y.add(W, -1.0)

            # compute (B-)Gram matrix for (X,Y)
            if std:
                s = numpy.sqrt(Y.dots(Y))
                Y.scale(s)
                s = numpy.sqrt(Y.dots(Y))
                XBY = Y.dot(X)
                YBY = Y.dot(Y)
            else:
                BY.select(Y.nvec())
                if not pro or nz == 0:
                    B(Y, BY)
                s = numpy.sqrt(BY.dots(Y))
                Y.scale(s)
                BY.scale(s)
                s = numpy.sqrt(BY.dots(Y))
                XBY = BY.dot(X)
                YBY = BY.dot(Y)
            YBX = conjugate(XBY)
            GB = numpy.concatenate((XBX, YBX))
            H = numpy.concatenate((XBY, YBY))
            GB = numpy.concatenate((GB, H), axis = 1)
    #        print('Gram matrix for (X,Y):')
    #        print(GB)

            # do pivoted Cholesky for GB
            U = GB
            ny = Y.nvec() # = nx
            ind, dropped, last_piv = piv_chol(U, nx, 1e-4)
            print(ind)
            print('dropped %d search directions' % dropped)
            
            # re-arrange/drop-linear-dependent search directions
            ny -= dropped
            nxy = nx + ny
            U = U[:nxy, :nxy]
            indy = ind[nx: nxy]
            for i in range(ny):
                indy[i] -= nx
            Y.copy(W, indy)
            W.select(ny)
            W.copy(Y)
            Y.select(ny)
            AY.select(ny)
            if not std:
                BY.copy(W, indy)
                W.copy(BY)
                BY.select(ny)
    
            # compute A-Gram matrix for (X,Y)
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

            # solve Rayleigh-Ritz eigenproblem
            GA = transform(GA, U)
            lmdxy, Q = sla.eigh(GA)
            print(lmdxy)
            Q = sla.solve_triangular(U, Q)

            # TODO: select new numbers of left and right eigenpairs
            if self.__lcon >= left:
                leftX_new = 0
                rightX_new = min(nxy, m)
            elif self.__rcon >= right:
                leftX_new = min(nxy, m)
                rightX_new = 0
            else:
                leftX_new = leftX
                rightX_new = rightX

            # compute RR coefficients for X and 'old search directions' Z
            # by re-arranging columns of Q
            QX = numpy.concatenate \
                ((Q[:, :leftX_new], Q[:, nxy - rightX_new:]), axis = 1)
            QZ = Q[:, leftX_new : nxy - rightX_new]
            lmdz = lmdxy[leftX_new : nxy - rightX_new]
            QXX = QX[:nx, :].copy()
            QYX = QX[nx:, :].copy()
            QXZ = QZ[:nx, :].copy()
            QYZ = QZ[nx:, :].copy()
    
            # X and 'old search directions' Z and their A- and B-images
            nx_new = leftX_new + rightX_new
            ix_new = 0
            nz = nxy - nx_new
            W.select(nx_new)
            Z.select(nx_new)
            AX.mult(QXX, W)
            AY.mult(QYX, Z)
            W.add(Z, 1.0)
            Z.select(nz)
            AY.mult(QYZ, Z)
            AZ = AY
            AZ.select(nz)
            AX.mult(QXZ, AZ)
            AX.select(nx_new, ix_new)
            W.copy(AX)
            AZ.add(Z, 1.0)
            if not std:
                Z.select(nx_new)
                BX.mult(QXX, W)
                BY.mult(QYX, Z)
                W.add(Z, 1.0)
                Z.select(nz)
                BY.mult(QYZ, Z)
                BZ = BY
                BZ.select(nz)
                BX.mult(QXZ, BZ)
                BX.select(nx_new, ix_new)
                W.copy(BX)
                BZ.add(Z, 1.0)
            else:
                BZ = Z
            Z.select(nx_new)
            X.mult(QXX, W) # W = X*QXX
            Y.mult(QYX, Z)
            W.add(Z, 1.0)
            Z.select(nz)
            X.mult(QXZ, Z) # Z = X*QXZ
            X.select(nx_new, ix_new)
            W.copy(X)      # X = W
            W.select(nz)
            Y.mult(QYZ, W) # Z += Y*QYZ
            Z.add(W, 1.0)

            nx = nx_new
            ix = ix_new
            leftX = leftX_new
            rightX = rightX_new
