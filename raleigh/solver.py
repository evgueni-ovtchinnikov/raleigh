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

def default_block_size(which, extra, init, threads):
    left = int(which[0])
    right = int(which[1])
    extra_left = int(extra[0])
    extra_right = int(extra[1])
    init_left = 0
    init_right = 0
    if init[0] is not None:
        init_left = int(init[0].nvec())
    if init[1] is not None:
        init_right = int(init[1].nvec())
    if threads <= 0:
        threads = 8
    if left == 0 and right == 0:
        return 0
    if left <= 0 and right <= 0:
        if init_left == 0 and init_right == 0:
            return 2*threads
        m = init_left + init_right
        m = threads*((m - 1)//threads + 1)
        if left < 0 or right < 0:
            m = max(m, 2*threads)
        return m
    left_total = 0
    right_total = 0
    if left > 0:
        if extra_left >= 0:
            left_total = max(left + extra_left, init_left)
        else:
            left_total = int(math.floor(max(left, init_left)*1.2))
    if right > 0:
        if extra_right >= 0:
            right_total = max(right + extra_right, init_right)
        else:
            right_total = int(math.floor(max(right, init_right)*1.2))
    if left < 0:
        left_total = right_total
    if right < 0:
        right_total = left_total
    m = int(left_total + right_total)
    m = threads*((m - 1)//threads + 1)
    if left < 0 or right < 0:
        m = max(m, 2*threads)
    return m

class DefaultConvergenceCriteria:
    def __init__(self, tol):
        self.tol_X = tol
    def satisfied(self, solver, i):
        cnv = solver.convergence_data('c', i)
        if cnv:
            return cnv
        err = solver.convergence_data('ve', i)
        return err[0] >= 0 and err[0] <= self.tol_X

class Options:
    def __init__(self):
        self.max_iter = 10
        self.block_size = -1
        self.threads = -1
        self.convergence_criteria = DefaultConvergenceCriteria(1e-3)
        self.verbosity = 0
        
class EstimatedErrors:
    def __init__(self):
        self.kinematic = numpy.ndarray((0,), dtype = numpy.float32)
        self.residual = numpy.ndarray((0,), dtype = numpy.float32)
    def __getitem__(self, item):
        return self.kinematic[item], self.residual[item]
    def append(self, est):
        self.kinematic = numpy.concatenate((self.kinematic, est[0,:]))
        self.residual = numpy.concatenate((self.residual, est[1,:]))
    def reorder(self, ind):
        self.kinematic = self.kinematic[ind]
        self.residual = self.residual[ind]

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
    def __init__(self, problem):
        self.__problem = problem
        self.__P = None

    def set_preconditioner(self, P):
        self.__P = P
        
    def convergence_data(self, what = 'converged', which = 0):
        if what[0] == 'c':
            return self.cnv[which]
        if what[0] == 'r':
            return self.res[which]
        if what[0 : 2] == 'va':
            return self.err_lmd[:, which]
        if what[0 : 2] == 've':
            return self.err_X[:, which]

    def solve \
        (self, eigenvectors, \
         options = Options(), \
         which = (-1,-1), \
         extra = (-1,-1), \
         init = (None, None)):
            
        verb = options.verbosity

        left = int(which[0])
        right = int(which[1])
        if left == 0 and right == 0:
            if verb > -1:
                print('No eigenpairs requested, quit')
            return

        m = int(options.block_size)
        if m < 0:
            m = default_block_size(which, extra, init, options.threads)
        elif m < 2:
            if verb > -1:
                print('Block size 1 too small, will use 2 instead')
            m = 2
        block_size = m
        self.block_size = m

        if left == 0:
            r = 0.0
        elif right == 0:
            r = 1.0
        elif left > 0 and right > 0:
            r = left/(left + 1.0*right)
        else:
            r = 0.5
        l = int(round(r*m))
        if l == 0 and r > 0.0:
            l = 1
        if l == m and r < 1.0:
            l = m - 1
        lr_ratio = r
        left_block_size = l
        
        extra_left = int(extra[0])
        extra_right = int(extra[1])
        if left >= 0:
            if extra_left >= 0:
                left_total = left + extra_left
            else:
                left_total = max(left, left_block_size)
            if verb > 0:
                print('left total: %d' % left_total)
        if right >= 0:
            if extra_right >= 0:
                right_total = right + extra_right
            else:
                right_total = max(right, block_size - left_block_size)
            if verb > 0:
                print('right_total: %d' % right_total)
        if verb > 0:
            print('left block size %d, right block size %d' % (l, m - l))

        # problem
        problem = self.__problem
        vector = problem.vector()
        problem_type = problem.type()
        std = (problem_type == 's')
        gen = (problem_type == 'g')
        pro = (problem_type == 'p')
        #self.__data_type = vector.data_type()

        #output
        self.lcon = 0
        self.rcon = 0
        self.eigenvalues = numpy.ndarray((0,), dtype = numpy.float64)
        self.eigenvalue_errors = EstimatedErrors()
        self.eigenvector_errors = EstimatedErrors()
        self.residual_norms = numpy.ndarray((0,), dtype = numpy.float32)

        # convergence data
        self.cnv = numpy.zeros((m,), dtype = numpy.int32)
        self.lmd = numpy.zeros((m,), dtype = numpy.float64)
        self.res = -numpy.ones((m,), dtype = numpy.float32)
        self.err_lmd = -numpy.ones((2, m,), dtype = numpy.float32)
        self.err_X = -numpy.ones((2, m,), dtype = numpy.float32)
        delta_res = 0.0

        # convergence history data
        dlmd = numpy.zeros((m, RECORDS), dtype = numpy.float32)
        dX = numpy.ones((m,), dtype = numpy.float32)
        acf = numpy.ones((2, m,), dtype = numpy.float32)

        # workspace
        X = vector.new_vectors(m)
        X.fill_random()
        #X.fill_orthogonal(m)
        Y = vector.new_vectors(m)
        Z = vector.new_vectors(m)
        W = vector.new_vectors(m)
        AX = vector.new_vectors(m)
        AY = vector.new_vectors(m)
        if not std:
            BX = vector.new_vectors(m)
            BY = vector.new_vectors(m)
        else:
            BX = X
            BY = Y

        # copy initial vectors if present
        l = left_block_size
        m = block_size
        init_lX = init[0]
        if init_lX is not None:
            init_left = min(l, init_lX.nvec())
            X.select(init_left)
            init_lX.select(init_left)
            init_lX.copy(X)
        else:
            init_left = 0
        init_rX = init[1]
        if init_rX is not None:
            init_right = min(m - l, init_rX.nvec())
            X.select(init_right, init_left)
            init_rX.select(init_right)
            init_rX.copy(X)

        # check for zero initial vectors
        X.select(m)
        s = X.dots(X)
        for i in range(m):
            if s[i] == 0.0:
                if verb > -1:
                    print('Zero initial guess, replacing with random')
                X.select(1, i)
                X.fill_random()
                s[i : i + 1] = X.dots(X)
        X.select(m)
        s = numpy.sqrt(X.dots(X))
        X.scale(s)
            
        # shorcuts
        lmd = self.lmd
        res = self.res
        A = self.__problem.A()
        B = self.__problem.B()
        P = self.__P
        err_lmd = self.err_lmd
        err_X = self.err_X

        # constraints (already available eigenvectors e.g. from previous run)
        Xc = eigenvectors
        nc = Xc.nvec()
        if not std:
            BXc = eigenvectors.clone()
            if nc > 0:
                B(Xc, BXc)
        else:
            BXc = Xc
        if nc > 0:
            Gc = BXc.dot(Xc)
            # approximate inverse of Gc
            Gci = 2*numpy.identity(nc) - Gc

        # initialize
        AZ = None
        BZ = None
        leftX = left_block_size
        rightX = block_size - leftX
        rec = 0
        ix = 0 # first X
        nx = block_size
        ny = block_size
        nz = 0

        if Xc.nvec() > 0:
            # orthogonalize X to Xc
            # std: X := X - Xc Xc* X
            # gen: X := X - Xc BXc* X
            # pro: X := X - Xc BXc* X
            Q = numpy.dot(Gci, X.dot(BXc))
            Xc.mult(Q, W)
            X.add(W, -1.0)

        if not std:
            B(X, BX)
        XBX = BX.dot(X)

        # do pivoted Cholesky for XBX to eliminate linear dependent X
        U = XBX
        ind, dropped, last_piv = piv_chol(U, 0, 1e-8)
        if dropped > 0:
            #print(ind)
            if verb > -1:
                print('dropped %d initial vectors out of %d' % (dropped, nx))
            # drop-linear-dependent initial vectors
            nx -= dropped
            if nx > 0:
                W.select(nx)
                X.copy(W, ind)
                W.copy(X)
            X.select(dropped, nx)
            X.fill_random()
            if not std:
                if nx > 0:
                    BX.copy(W, ind)
                    W.copy(BX)
                BX.select(dropped, nx)
                B(X, BX)
            if Xc.nvec() > 0:
                # orthogonalize X to Xc
                Q = numpy.dot(Gci, X.dot(BXc))
                Xc.mult(Q, W)
                X.add(W, -1.0)
                if not std:
                    BXc.mult(Q, W)
                    BX.add(W, -1.0)
            nx = m
            X.select(nx)
            if not std:
                BX.select(nx)
            XBX = BX.dot(X)
            
#        lmdB, Q = sla.eigh(XBX)
#        print(lmdB)
            
        # Rayleigh-Ritz in the initial space
        if pro:
            A(BX, AX)
            XAX = AX.dot(BX)
        else:
            A(X, AX)
            XAX = AX.dot(X)
        lmd_in, Q = sla.eigh(XAX, XBX, overwrite_a = True, overwrite_b = True)
        W.select(m)
        X.mult(Q, W)
        W.copy(X)
        AX.mult(Q, W)
        W.copy(AX)
        if not std:
            BX.mult(Q, Z)
            Z.copy(BX)

        # main CG loop
        for self.iteration in range(options.max_iter):

            if verb > 0:
                print('------------- iteration %d' % self.iteration)

            if pro:
                XAX = AX.dot(BX)
            else:
                XAX = AX.dot(X)
            XBX = BX.dot(X)
            da = XAX.diagonal()
            db = XBX.diagonal()
            new_lmd = da/db
            if self.iteration > 0:
                # compute eigenvalue decrements
                for i in range(nx):
                    if dlmd[ix + i, rec - 1] == 0: # previous data not available
                        continue
                    delta = lmd[ix + i] - new_lmd[i]
                    eps = 1e-4*max(abs(new_lmd[i]), abs(lmd[ix + i]))
                    if abs(delta) > eps:
                        dlmd[ix + i, rec - 1] = delta
#                print(dlmd[ix : ix + nx, rec - 1])
                if verb > 3:
                    print('eigenvalues shifts history:')
                    print(numpy.array_str(dlmd[ix : ix + nx, :rec].T, \
                                          precision = 2))
            lmd[ix : ix + nx] = new_lmd
            
            # estimate error in residual computation due to the error in
            # computing AX, to be used in detecting convergence stagnation
            H = abs(XAX - conjugate(XAX))
            delta = numpy.amax(H)
            if gen:
                s = numpy.sqrt(X.dots(X))
                delta /= numpy.amax(s)
            elif pro:
                s = numpy.sqrt(BX.dots(BX))
                delta /= numpy.amax(s)
            delta_res = max(delta_res, delta)
            if verb > 1:
                print('estimated error in residual: %e' % delta_res)
            
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

            if Xc.nvec() > 0:
                # orthogonalize W to Xc
                # std: W := W - Xc Xc* W
                # gen: W := W - BXc Xc* W
                # pro: W := W - Xc BXc* W
                if pro:
                    Q = numpy.dot(Gci, W.dot(BXc))
                else:
                    Q = numpy.dot(Gci, W.dot(Xc))
                if gen:
                    BXc.mult(Q, Y)
                else:
                    Xc.mult(Q, Y)
                W.add(Y, -1.0)
            s = W.dots(W)
            res[ix : ix + nx] = numpy.sqrt(s)
    
            # kinematic error estimates
            if rec > 3: # sufficient history available
                for i in range(nx):
                    if dX[ix + i] > 0.01:
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
                    acf[1, ix + i] = acf[0, ix + i]
                    acf[0, ix + i] = qi # a.c.f. estimate
                    # esimate error based on a.c.f.
                    theta = qi/(1 - qi)
                    d = theta*dlmd[ix + i, rec - 1]
                    err_lmd[0, ix + i] = d
                    qx = math.sqrt(qi)
                    err_X[0, ix + i] = dX[ix + i]*qx/(1 - qx)

            # residual-based error estimates:
            # asymptotic Lehmann for eigenvalues
            # generalized (extended gap) Davis-Kahan for eigenvectors
            l = 0
            for k in range(1, leftX):
                i = ix + k
                if dX[i] > 0.01:
                    break
                if lmd[i] - lmd[i - 1] > res[i]:
                    l = k
            if l > 0:
                i = ix + l
                t = lmd[i]
                if verb > 1:
                    print('using left pole at lmd[%d] = %e' % (i, t))
                m = block_size
                for k in range(l):
                    i = ix + k
                    s = res[i]
                    err_lmd[1, i] = s*s/(t - lmd[i])
                    err_X[1, i] = s/(t - lmd[i])
            l = 0
            for k in range(1, rightX):
                i = ix + nx - k - 1
                if dX[i] > 0.01:
                    break
                if lmd[i + 1] - lmd[i] > res[i]:
                    l = k
            if l > 0:
                i = ix + nx - l - 1
                t = lmd[i]
                if verb > 1:
                    print('using right pole at lmd[%d] = %e' % (i, t))
                m = block_size
                for k in range(l):
                    i = ix + nx - k - 1
                    s = res[i]
                    err_lmd[1, i] = s*s/(lmd[i] - t)
                    err_X[1, i] = s/(lmd[i] - t)

            if verb > 1:
                msg = 'eigenvalue   residual  ' + \
                'estimated errors (kinematic/residual)' + \
                '  a.c.f.'
                print(msg)
                for i in range(block_size):
                    print('%e %.1e  %.1e / %.1e    %.1e / %.1e  %.1e' % \
                          (lmd[i], res[i], \
                          abs(err_lmd[0, i]), abs(err_lmd[1, i]), \
                          abs(err_X[0, i]), abs(err_X[1, i]), acf[0, i]))

            lcon = 0
            for i in range(leftX):
                j = self.lcon + i
                k = ix + i
                if res[k] < 10*delta_res and acf[0, k] > acf[1, k]:
                    if verb > -1:
                        msg = 'left eigenvector %d stagnated,' + \
                        ' eigenvalue %e, error %e / %e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    lcon += 1
                    self.cnv[k] = -1
                elif options.convergence_criteria.satisfied(self, k):
                    if verb > 0:
                        msg = 'left eigenvector %d converged,' + \
                        ' eigenvalue %e, error %e / %e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    lcon += 1
                    self.cnv[k] = 1
                else:
                    break
            rcon = 0
            for i in range(rightX):
                j = self.rcon + i
                k = ix + nx - i - 1
                if res[k] < 10*delta_res and acf[0, k] > acf[1, k]:
                    if verb > -1:
                        msg = 'right eigenvector %d stagnated,' + \
                        ' eigenvalue %e, error %e / %e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    rcon += 1
                    self.cnv[k] = -1
                elif options.convergence_criteria.satisfied(self, k):
                    if verb > 0:
                        msg = 'right eigenvector %d converged, \n' + \
                        ' eigenvalue %e, error %e / %e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    rcon += 1
                    self.cnv[k] = 1
                else:
                    break

            # move converged X to Xc, update Gram matrix for Xc
            ncon = Xc.nvec()
            if lcon > 0:
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[ix : ix + lcon]))
                self.eigenvalue_errors.append(err_lmd[:, ix : ix + lcon])
                self.eigenvector_errors.append(err_X[:, ix : ix + lcon])
                self.residual_norms = numpy.concatenate \
                    ((self.residual_norms, res[ix : ix + lcon]))
                X.select(lcon, ix)
                if std and ncon > 0:
                    if ncon > 0:
                        Gu = X.dot(Xc)
                Xc.append(X)
                if not std:
                    if ncon > 0:
                        Gu = X.dot(BXc)
                    BX.select(lcon, ix)
                    BXc.append(BX)
                    if ncon < 1:
                        Gc = BXc.dot(Xc)
                    else:
                        Gl = BXc.dot(X)
                else:
                    if ncon < 1:
                        Gc = Xc.dot(Xc)
                    else:
                        Gl = Xc.dot(X)
                if ncon > 0:
                    Gc = numpy.concatenate((Gc, Gu), axis = 1)
                    Gc = numpy.concatenate((Gc, Gl))
                ncon += lcon
            if rcon > 0:
                jx = ix + nx
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[jx - rcon : jx]))
                self.eigenvalue_errors.append(err_lmd[:, jx - rcon : jx])
                self.eigenvector_errors.append(err_X[:, jx - rcon : jx])
                self.residual_norms = numpy.concatenate \
                    ((self.residual_norms, res[jx - rcon : jx]))
                X.select(rcon, jx - rcon)
                if std and ncon > 0:
                    if ncon > 0:
                        Gu = X.dot(Xc)
                Xc.append(X)
                if not std:
                    if ncon > 0:
                        Gu = X.dot(BXc)
                    BX.select(rcon, jx - rcon)
                    BXc.append(BX)
                    if ncon < 1:
                        Gc = BXc.dot(Xc)
                    else:
                        Gl = BXc.dot(X)
                else:
                    if ncon < 1:
                        Gc = Xc.dot(Xc)
                    else:
                        Gl = Xc.dot(X)
                if ncon > 0:
                    Gc = numpy.concatenate((Gc, Gu), axis = 1)
                    Gc = numpy.concatenate((Gc, Gl))
                ncon += rcon
            if ncon > 0:
                H = Gc - numpy.identity(ncon)
                if verb > 1:
                    print('Gram error: %e' % nla.norm(H))
                # approximate inverse, good enough for Gram error up to 1e-8
                Gci = 2*numpy.identity(ncon) - Gc

            self.lcon += lcon
            self.rcon += rcon
            left_converged = left >= 0 and self.lcon >= left
            right_converged = right >= 0 and self.rcon >= right
            if left_converged and right_converged:
                break
            
            leftX -= lcon
            rightX -= rcon
            
            ## re-select Xs, AXs, BXs accordingly
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
                Num = ZAY - numpy.dot(ZBY, numpy.diag(lmd[iy : iy + ny]))
                ny = Y.nvec()
                Lmd = numpy.ndarray((1, ny))
                Mu = numpy.ndarray((nz, 1))
                Lmd[0, :] = lmd[iy : iy + ny]
                Mu[:, 0] = lmdz
                Den = Mu - Lmd
                # TODO: avoid division by zero
                Beta = Num/Den
                
                # conjugate search directions
                AZ.select(ny)
                Z.mult(Beta, AZ)
                Y.add(AZ, -1.0)
                if pro: # if gen or nz == 0, BY computed later
                    BZ.mult(Beta, AZ)
                    W.add(AZ, -1.0)
                    BY.select(ny)
                    W.copy(BY)

            if Xc.nvec() > 0: #and (P is not None or gen):
                # orthogonalize Y to Xc
                # std: W := W - Xc Xc* W (not needed if P is None)
                # gen: W := W - Xc BXc* W
                # pro: W := W - Xc BXc* W (not needed if P is None)
                if not std:
                    Q = numpy.dot(Gci, Y.dot(BXc))
                else:
                    Q = numpy.dot(Gci, Y.dot(Xc))
                Xc.mult(Q, W)
                Y.add(W, -1.0)

            # compute (B-)Gram matrix for (X,Y)
            if std:
                s = numpy.sqrt(Y.dots(Y))
                Y.scale(s)
                s = numpy.sqrt(Y.dots(Y))
                if nx > 0:
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
                if nx > 0:
                    XBY = BY.dot(X)
                YBY = BY.dot(Y)

            if nx > 0:
                YBX = conjugate(XBY)
                GB = numpy.concatenate((XBX, YBX))
                H = numpy.concatenate((XBY, YBY))
                GB = numpy.concatenate((GB, H), axis = 1)
            else:
                GB = YBY

            # do pivoted Cholesky for GB
            U = GB
            ny = Y.nvec()
            ind, dropped, last_piv = piv_chol(U, nx, 1e-4)
            if dropped > 0:
                #print(ind)
                if verb > -1:
                    print('dropped %d search directions out of %d' \
                          % (dropped, ny))
            
            # re-arrange/drop-linear-dependent search directions
            ny -= dropped
            nxy = nx + ny
            U = U[:nxy, :nxy]
            indy = ind[nx: nxy]
            for i in range(ny):
                indy[i] -= nx
            W.select(ny)
            Y.copy(W, indy)
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
                if nx > 0:
                    XAY = AY.dot(BX)
                YAY = AY.dot(BY)
            else:
                A(Y, AY)
                if nx > 0:
                    XAY = AY.dot(X)
                YAY = AY.dot(Y)
            if nx > 0:
                YAX = conjugate(XAY)
                GA = numpy.concatenate((XAX, YAX))
                H = numpy.concatenate((XAY, YAY))
                GA = numpy.concatenate((GA, H), axis = 1)
            else:
                GA = YAY

            # solve Rayleigh-Ritz eigenproblem
            G = transform(GA, U)
            YAY = G[nx : nxy, nx : nxy]
            lmdy, Qy = sla.eigh(YAY)
            G[:, nx : nxy] = numpy.dot(G[:, nx : nxy], Qy)
            if nx > 0:
                G[nx : nxy, :nx] = conjugate(G[:nx, nx : nxy])
            G[nx : nxy, nx : nxy] = numpy.dot(conjugate(Qy), G[nx : nxy, nx : nxy])
            
            lmdxy, Q = sla.eigh(G)
            #print(lmdxy)

            if lcon + rcon > 0:
                if left < 0:
                    shift_left_max = lcon
                else:
                    shift_left_max = max(0, left_total - self.lcon - leftX)
                if right < 0:
                    shift_right_max = rcon
                else:
                    shift_right_max = max(0, right_total - self.rcon - rightX)
                if lcon + rcon <= ny:
                    shift_left = lcon
                    shift_right = rcon
                else:
                    shift_left = min(lcon, int(round(lr_ratio*ny)))
                    shift_right = min(rcon, ny - shift_left)
                shift_left = min(shift_left, shift_left_max)
                shift_right = min(shift_right, shift_right_max)
            else:
                shift_left = 0
                shift_right = 0

            # select new numbers of left and right eigenpairs
            if left > 0 and lcon > 0 and self.lcon >= left: # left side converged
                if verb > 0:
                    print('left side converged')
                leftX_new = 0
                rightX_new = rightX + shift_right
                shift_left = -leftX
                left_block_size_new = ix + leftX + rightX - rightX_new
                lr_ratio = 0.0
                ix_new = left_block_size_new
            elif right > 0 and rcon > 0 and self.rcon >= right: # right side converged
                if verb > 0:
                    print('right side converged')
                ix_new = ix - shift_left
                leftX_new = min(nxy, block_size - ix_new)
                rightX_new = 0
                shift_right = -rightX
                left_block_size_new = ix + leftX + rightX
                lr_ratio = 1.0
            else:
                leftX_new = leftX + shift_left
                rightX_new = rightX + shift_right
                left_block_size_new = left_block_size
                ix_new = ix - shift_left
            nx_new = leftX_new + rightX_new
            #print('shifts:', shift_left, shift_right)
            if verb > 2:
                print('left X: %d %d' % (leftX, leftX_new))
                print('right X: %d %d' % (rightX, rightX_new))
                print('new ix: %d, nx: %d, nxy: %d' % (ix_new, nx_new, nxy))

            # shift eigenvalues etc.
            m = block_size
            l = left_block_size
            nl = left_block_size_new
            cnv = self.cnv
            if shift_left > 0:
                for i in range(l - shift_left):
                    cnv[i] = cnv[i + shift_left]
                    lmd[i] = lmd[i + shift_left]
                    res[i] = res[i + shift_left]
                    acf[:, i] = acf[:, i + shift_left]
                    err_lmd[:, i] = err_lmd[:, i + shift_left]
                    dlmd[i, :] = dlmd[i + shift_left, :]
                    err_X[:, i] = err_X[:, i + shift_left]
                    dX[i] = dX[i + shift_left]
            if shift_left >= 0:
                for i in range(l - shift_left, nl):
                    cnv[i] = 0
                    res[i] = -1.0
                    acf[:, i] = 1.0
                    err_lmd[:, i] = -1.0
                    dlmd[i, :] = 0
                    err_X[:, i] = -1.0
                    dX[i] = 0
            if shift_right > 0:
                for i in range(m - 1, l + shift_right - 1, -1):
                    cnv[i] = cnv[i - shift_right]
                    lmd[i] = lmd[i - shift_right]
                    res[i] = res[i - shift_right]
                    acf[:, i] = acf[:, i - shift_right]
                    err_lmd[:, i] = err_lmd[:, i - shift_right]
                    dlmd[i, :] = dlmd[i - shift_right, :]
                    err_X[:, i] = err_X[:, i - shift_right]
                    dX[i] = dX[i - shift_right]
            if shift_right >= 0:
                for i in range(l + shift_right - 1, nl - 1, -1):
                    cnv[i] = 0
                    res[i] = -1.0
                    acf[:, i] = 1.0
                    err_lmd[:, i] = -1.0
                    dlmd[i, :] = 0
                    err_X[:, i] = -1.0
                    dX[i] = 0

            # estimate changes in eigenvalues and eigenvectors
            lmdx = numpy.concatenate \
                ((lmdxy[:leftX_new], lmdxy[nxy - rightX_new:]))
            QX = numpy.concatenate \
                ((Q[:, :leftX_new], Q[:, nxy - rightX_new:]), axis = 1)
            QYX = QX[nx:, :].copy()
            lmdX = numpy.ndarray((1, nx_new))
            lmdY = numpy.ndarray((ny, 1))
            lmdX[0, :] = lmdx
            lmdY[:, 0] = lmdy
            Delta = (lmdY - lmdX)*QYX*QYX
            dX[ix_new : ix_new + nx_new] = nla.norm(QYX, axis = 0)
            if rec == RECORDS:
                for i in range(rec - 1):
                    dlmd[:, i] = dlmd[:, i + 1]
            else:
                rec += 1
            dlmd[ix_new : ix_new + nx_new, rec - 1] = numpy.sum(Delta, axis = 0)

            # compute RR coefficients for X and 'old search directions' Z
            # by re-arranging columns of Q
            Q[nx : nxy, :] = numpy.dot(Qy, Q[nx : nxy, :])
            Q = sla.solve_triangular(U, Q)
            QX = numpy.concatenate \
                ((Q[:, :leftX_new], Q[:, nxy - rightX_new:]), axis = 1)
            lft = max(leftX, leftX_new)
            rgt = max(rightX, rightX_new)
            nz = nxy - lft - rgt
            lmdz = lmdxy[lft : nxy - rgt]
            QZ = Q[:, lft : nxy - rgt]
            if nx > 0:
                QXX = QX[:nx, :].copy()
            QYX = QX[nx:, :].copy()
            if nx > 0:
                QXZ = QZ[:nx, :].copy()
            QYZ = QZ[nx:, :].copy()
    
            # X and 'old search directions' Z and their A- and B-images
            W.select(nx_new)
            Z.select(nx_new)
            if nx > 0:
                AX.mult(QXX, W)
                AY.mult(QYX, Z)
                W.add(Z, 1.0) # W = AX*QXX + AY*QYX
            else:
                AY.mult(QYX, W)
            Z.select(nz)
            if nz > 0:
                AY.mult(QYZ, Z)
                AZ = AY
                AZ.select(nz)
                if nx > 0:
                    AX.mult(QXZ, AZ)
                else:
                    AZ.zero()
            AX.select(nx_new, ix_new)
            W.copy(AX)
            if nz > 0:
                AZ.add(Z, 1.0) # Z = AX*QXZ + AY*QYZ
            if not std:
                Z.select(nx_new)
                if nx > 0:
                    BX.mult(QXX, W)
                    BY.mult(QYX, Z)
                    W.add(Z, 1.0)
                else:
                    BY.mult(QYX, W)
                Z.select(nz)
                if nz > 0:
                    BY.mult(QYZ, Z)
                    BZ = BY
                    BZ.select(nz)
                    if nx > 0:
                        BX.mult(QXZ, BZ)
                    else:
                        BZ.zero()
                BX.select(nx_new, ix_new)
                W.copy(BX)
                if nz > 0:
                    BZ.add(Z, 1.0)
            else:
                BZ = Z
            Z.select(nx_new)
            if nx > 0:
                X.mult(QXX, W)
                Y.mult(QYX, Z)
                W.add(Z, 1.0) # W = X*QXX + Y*QYX
            else:
                Y.mult(QYX, W)
            Z.select(nz)
            if nz > 0:
                if nx > 0:
                    X.mult(QXZ, Z)
                else:
                    Z.zero()
            X.select(nx_new, ix_new)
            W.copy(X)
            if nz > 0:
                W.select(nz)
                Y.mult(QYZ, W)
                Z.add(W, 1.0) # Z = X*QXZ + Y*QYZ
                
            nx = nx_new
            ix = ix_new
            leftX = leftX_new
            rightX = rightX_new
            left_block_size = left_block_size_new
