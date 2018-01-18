'''
RAL EIGensolver for real symmetric and Hermitian problems.

'''

from raleigh.piv_chol import piv_chol
import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla

RECORDS = 20

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

class Solver:
    def __init__ \
        (self, problem, eigenvectors, \
         options = Options(), which = (-1,-1), extra = (-1,-1)):

        vector = problem.vector()
        self.__problem = problem
        problem_type = problem.type()
        std = (problem_type == 's')
        self.__data_type = vector.data_type()

        left = int(which[0])
        right = int(which[1])
        if left == 0 and right == 0:
            raise ValueError('wrong number of needed eigenpairs')
        extra_left = extra[0]
        extra_right = extra[1]
        if extra_left < 0:
            extra_left = 0 # TODO: set proper default
        if extra_right < 0:
            extra_right = 0 # TODO: set proper default
        self.__left = left
        self.__right = right
        self.__left_total = left + extra_left
        self.__right_total = right + extra_right

        m = int(options.block_size)
        if m < 0:
            if left < 0 or right < 0:
                m = 16
            else:
                m = max(2, left + right)
        mm = m + m
        self.__block_size = m

        if left == 0:
            q = 0.0
        elif right == 0:
            q = 1.0
        elif left > 0 and right > 0:
            q = left/(left + 1.0*right)
        else:
            q = 0.5
        l = int(round(q*m))
        if l == 0 and q > 0.0:
            l = 1
        if l == m and q < 1.0:
            l = m - 1
        self.__lr_ratio = q
        self.__left_block_size = l
        print('left block size %d, right block size %d' % (l, m - l))

        self.eigenvalues = numpy.ndarray((0,), dtype = numpy.float64)
        self.lmd = numpy.ndarray((m,), dtype = numpy.float64)
        self.res = numpy.ndarray((m,), dtype = numpy.float64)
        self.__err_est = options.err_est
        self.__res_tol = options.res_tol
        self.__max_iter = options.max_iter
        self.__Xc = eigenvectors
        if not std:
            self.__BXc = eigenvectors.clone()
        self.__ind = numpy.ndarray((m,), dtype = numpy.int32)
        self.__lmd = numpy.ndarray((mm,), dtype = numpy.float64)
        self.__dlmd = numpy.zeros((m, RECORDS), dtype = numpy.float64)
        self.__dX = numpy.ones((m,), dtype = numpy.float64)
        self.__q = numpy.ones((mm,), dtype = numpy.float64)
        self.__err_lmd = -numpy.ones((mm,), dtype = numpy.float64)
        self.__err_X = -numpy.ones((mm,), dtype = numpy.float64)
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
        std = (problem_type == 's')
        gen = (problem_type == 'g')
        pro = (problem_type == 'p')
        lmd = self.lmd
        res = self.res
        A = self.__problem.A()
        B = self.__problem.B()
        P = self.__P
        block_size = self.__block_size
        left_block_size = self.__left_block_size
        left = self.__left
        right = self.__right
        left_total = self.__left_total
        right_total = self.__right_total
        dlmd = self.__dlmd
        dX = self.__dX
        q = self.__q
        err_lmd = self.__err_lmd
        err_X = self.__err_X
        Xc = self.__Xc
        if not std:
            BXc = self.__BXc
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
        leftX = left_block_size
        rightX = block_size - leftX
        lr_ratio = self.__lr_ratio
        lconv = 0
        rconv = 0
        rec = 0
        ix = 0 # first X
        nx = block_size
        ny = block_size
        nz = 0
        res_tol = self.__res_tol

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
        lmd_in, Q = sla.eigh(XAX, XBX, overwrite_a = True, overwrite_b = True)
        X.mult(Q, W)
        W.copy(X)
        AX.mult(Q, W)
        W.copy(AX)
        if not std:
            BX.mult(Q, Z)
            Z.copy(BX)

        for iteration in range(self.__max_iter):
            print('------------- iteration %d' % iteration)
            if pro:
                XAX = AX.dot(BX)
            else:
                XAX = AX.dot(X)
            XBX = BX.dot(X)
            da = XAX.diagonal()
            db = XBX.diagonal()
            print(da)
            print(db)
            new_lmd = da/db
            if iteration > 0:
                # compute eigenvalue decrements
                for i in range(nx):
                    if dlmd[ix + i, rec - 1] == 0: # previous data not available
                        continue
                    delta = lmd[ix + i] - new_lmd[i]
                    eps = 1e-6*max(abs(new_lmd[i]), abs(lmd[ix + i]))
                    if abs(delta) > eps:
                        dlmd[ix + i, rec - 1] = delta
#                    print(dlmd[ix + i, rec - 1], lmd[ix + i] - new_lmd[i], dX[ix + i])
                print(numpy.array_str(dlmd[ix : ix + nx, :rec].T, precision = 2))
            lmd[ix : ix + nx] = new_lmd
            
            # compute residuals
            # std: A X - X lmd
            # gen: A X - B X lmd
            # pro: A B X - X lmd
            W.select(nx, ix)
            Y.select(nx)
            AX.copy(W)
            if gen:
                W.add(BX, -lmd[ix : ix + nx])
            else:
                W.add(X, -lmd[ix : ix + nx])
            s = W.dots(W)
            print(numpy.sqrt(s))

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
            res[ix : ix + nx] = numpy.sqrt(s)
    
            ## TODO: estimate errors
            if rec > 3: # sufficient history available
                for i in range(nx):
                    if dX[ix + i] > 0.1:
                        continue
                    k = 0
                    s = 0
                    # go through the last 1/3 of the history
                    for r in range(rec - 1, rec - rec//3 - 2, -1):
                        d = dlmd[ix + i, r]
                        if d == 0:
                            break
                        k = k + 1
                        s = s + d
                    if k < 2 or s == 0:
                        continue
                    # estimate asymptotic convergence factor (a.c.f)
                    qi = dlmd[ix + i, rec - 1]/s
                    if qi <= 0:
                        continue
                    qi = qi**(1.0/(k - 1))
                    q[ix + i] = qi # a.c.f. estimate
                    # esimate error based on a.c.f.
                    theta = qi/(1 - qi)
                    d = theta*dlmd[ix + i, rec - 1]
                    err_lmd[ix + i] = d
                    qx = math.sqrt(qi)
                    err_X[ix + i] = dX[ix + i]*qx/(1 - qx)

            print('eigenvalue   residual     errors       a.c.f.')
            for i in range(ix, ix + nx):
                print('%e %.1e  %.1e %.1e  %.1e' % \
                      (lmd[i], res[i], abs(err_lmd[i]), err_X[i], q[i]))

            lcon = 0
            for i in range(leftX):
                j = lconv + i
                if res[ix + i] < res_tol:
                    print('left eigenvector %d converged' % j)
                    lcon += 1
                else:
                    break
            lconv += lcon
            rcon = 0
            for i in range(rightX):
                j = rconv + i
                if res[ix + nx - i - 1] < res_tol:
                    print('right eigenvector %d converged' % j)
                    rcon += 1
                else:
                    break
            rconv += rcon

            ## move converged X to Xc,
            if lcon > 0:
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[ix : ix + lcon]))
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
            left_converged = left >= 0 and lconv >= left
            right_converged = right >= 0 and rconv >= right
            if left_converged and right_converged:
                break
            
            leftX -= lcon
            rightX -= rcon
            
            ## select Xs, AXs, BXs accordingly
            iy = ix
            ny = nx
            ix += lcon
            nx -= lcon + rcon
            X.select(nx, ix)
            AX.select(nx, ix)
            if not std:
                BX.select(nx, ix)
            XAX = XAX[lcon : lcon + nx, lcon : lcon + nx]
            XBX = XBX[lcon : lcon + nx, lcon : lcon + nx]
#            print(ix, nx, XBX.shape)

            if P is None:
                W.copy(Y)
            else:
                P(W, Y)
            
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
#                print(ZAY.shape, ZBY.shape, ny, numpy.diag(lmd[iy : iy + ny]).shape)
                Num = ZAY - numpy.dot(ZBY, numpy.diag(lmd[iy : iy + ny]))
                ny = Y.nvec()
                Lmd = numpy.ndarray((1, ny))
                Mu = numpy.ndarray((nz, 1))
                Lmd[0, :] = lmd[iy : iy + ny]
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
 #           print(nx, X.nvec(), Y.nvec(), XBX.shape, YBX.shape)
            GB = numpy.concatenate((XBX, YBX))
            H = numpy.concatenate((XBY, YBY))
            GB = numpy.concatenate((GB, H), axis = 1)

            # do pivoted Cholesky for GB
            U = GB
            ny = Y.nvec()
            ind, dropped, last_piv = piv_chol(U, nx, 1e-4)
            if dropped > 0:
                #print(ind)
                print('dropped %d search directions out of %d' % (dropped, ny))
            
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
            # and estimate eigenvalue and eigenvector shifts
            G = transform(GA, U)
            YAY = G[nx : nxy, nx : nxy]
            lmdy, Qy = sla.eigh(YAY)
            G[:, nx : nxy] = numpy.dot(G[:, nx : nxy], Qy)
            G[nx : nxy, :nx] = conjugate(G[:nx, nx : nxy])
            G[nx : nxy, nx : nxy] = numpy.dot(conjugate(Qy), G[nx : nxy, nx : nxy])
            
            # estimate eigenvalue and eigenvector shifts
            Num = G[:nx, nx : nxy]
            Num = numpy.absolute(Num)
            Lmd = numpy.ndarray((1, ny))
            Mu = numpy.ndarray((nx, 1))
            Lmd[0, :] = lmdy
            Mu[:, 0] = lmd[ix : ix + nx]
            Den = Mu - Lmd
            # safeguard against division overflow
            eps = 1e-16
            exclude = numpy.absolute(Den) < eps*Num
            Den[exclude] = eps*Num[exclude]
            dX[ix : ix + nx] = nla.norm(Num/Den, axis = 1)
            Num = Num*Num
            if rec == RECORDS:
                for i in range(rec - 1):
                    dlmd[:, i] = dlmd[:, i + 1]
            else:
                rec += 1
            dlmd[ix : ix + nx, rec - 1] = -numpy.sum(Num/Den, axis = 1)

            lmdxy, Q = sla.eigh(G)
            Q[nx : nxy, :] = numpy.dot(Qy, Q[nx : nxy, :])
            Q = sla.solve_triangular(U, Q)
            #print(lmdxy)

            # TODO: compute proper shifts in the case of known number of
            # wanted eigenpairs
            if lcon + rcon > 0:
                shift_left_max = max(0, left_total - lconv - leftX)
                shift_right_max = max(0, right_total - rconv - rightX)
                if lcon + rcon <= ny:
                    shift_left = lcon
                    shift_right = rcon
                else:
                    shift_left = min(lcon, int(round(lr_ratio*ny)))
                    shift_right = min(rcon, ny - shift_left)
                print(shift_left, shift_left_max)
                print(shift_right, shift_right_max)
                # TODO: debug changing ix
                shift_left = min(shift_left, shift_left_max)
                shift_right = min(shift_right, shift_right_max)
                #print(ny, shift_left, nxy)
            else:
                shift_left = 0
                shift_right = 0

            # select new numbers of left and right eigenpairs
            if left > 0 and lcon > 0 and lconv >= left: # left side converged
                leftX_new = 0
                rightX_new = rightX + shift_right #min(nxy, block_size)
                shift_left = -leftX
                left_block_size_new = block_size - rightX_new
                lr_ratio = 0.0
                ix_new = left_block_size_new
            elif right > 0 and rcon > 0 and rconv >= right: # right side converged
                leftX_new = leftX + shift_left #min(nxy, block_size)
                rightX_new = 0
                shift_right = -rightX
                left_block_size_new = leftX_new
                lr_ratio = 1.0
                ix_new = ix - shift_left
            else:
                leftX_new = leftX + shift_left
                rightX_new = rightX + shift_right
                left_block_size_new = left_block_size
                ix_new = ix - shift_left
            nx_new = leftX_new + rightX_new
            print('left X: %d %d' % (leftX, leftX_new))
            print('right X: %d %d' % (rightX, rightX_new))
            print('new ix: %d, nx: %d' % (ix_new, nx_new))

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
            print(W.selected())
            print(AX.selected())
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
            X.mult(QXX, W)
            Y.mult(QYX, Z)
            W.add(Z, 1.0) # W = X*QXX + Y*QYX
            Z.select(nz)
            X.mult(QXZ, Z)
            X.select(nx_new, ix_new)
            W.copy(X)
            W.select(nz)
            Y.mult(QYZ, W)
            Z.add(W, 1.0) # Z = X*QXZ + Y*QYZ

            # re-arrange eigenvalues, shifts etc.
            m = block_size
            l = left_block_size
            nl = left_block_size_new
            if shift_left > 0:
                for i in range(l - shift_left):
                    lmd[i] = lmd[i + shift_left]
                    dlmd[i, :] = dlmd[i + shift_left, :]
                    dX[i] = dX[i + shift_left]
                    q[i] = q[i + shift_left]
                    q[m + i] = q[m + i + shift_left]
                    err_lmd[i] = err_lmd[i + shift_left]
                    err_lmd[m + i] = err_lmd[m + i + shift_left]
            if shift_left >= 0:
                for i in range(l - shift_left, nl):
                    dlmd[i, :] = 0
                    dX[i] = 0
                    q[i] = 1.0
                    q[m + i] = 1.0
                    err_lmd[i] = -1.0
                    err_lmd[m + i] = -1.0
            if shift_right > 0:
                for i in range(m - 1, l + shift_right - 1, -1):
                    lmd[i] = lmd[i - shift_right]
                    dlmd[i, :] = dlmd[i - shift_right, :]
                    dX[i] = dX[i - shift_right]
                    q[i] = q[i - shift_right]
                    q[m + i] = q[m + i - shift_right]
                    err_lmd[i] = err_lmd[i - shift_right]
                    err_lmd[m + i] = err_lmd[m + i - shift_right]
            if shift_right >= 0:
                for i in range(l + shift_right - 1, nl - 1, -1):
                    dlmd[i, :] = 0
                    dX[i] = 0
                    q[i] = 1.0
                    q[m + i] = 1.0
                    err_lmd[i] = -1.0
                    err_lmd[m + i] = -1.0

            nx = nx_new
            ix = ix_new
            leftX = leftX_new
            rightX = rightX_new
            left_block_size = left_block_size_new
            #print(ix, leftX, rightX, left_block_size)
