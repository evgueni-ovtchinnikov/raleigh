'''
RAL EIGensolver for real symmetric and Hermitian problems.

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla

RECORDS = 100

class DefaultConvergenceCriteria:
    '''
    Convergence criteria to be used if not specified by the user via Options
    (see Options.convergence_criteria below).
    '''
    def __init__(self):
        self.tolerance = 1e-3
        self.error = 'kinematic eigenvector error'
    def set_error_tolerance(self, error, tolerance):
        self.error = error
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data(self.error, i)
        return err >= 0 and err <= self.tolerance

class Options:
    '''
    Solver options.
    '''
    def __init__(self):
        self.verbosity = 0
        self.max_iter = -1
        self.min_iter = 0
        self.block_size = -1
        self.threads = -1
        self.convergence_criteria = None
        self.stopping_criteria = None
        self.detect_stagnation = True
        self.max_quota = 0.5

class EstimatedErrors:
    '''
    Estimated errors accumulator.
    '''
    def __init__(self):
        self.kinematic = numpy.ndarray((0,), dtype=numpy.float32)
        self.residual = numpy.ndarray((0,), dtype=numpy.float32)
    def __getitem__(self, item):
        return self.kinematic[item], self.residual[item]
    def append(self, est):
        self.kinematic = numpy.concatenate((self.kinematic, est[0, :]))
        self.residual = numpy.concatenate((self.residual, est[1, :]))
    def reorder(self, ind):
        self.kinematic = self.kinematic[ind]
        self.residual = self.residual[ind]

class Problem:
    '''
    Eigenvalue problem specification.
    '''
    def __init__(self, v, A, B=None, prod=None):
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
    '''
    Eigenvalue problem solver specification.
    '''
    def __init__(self, problem):
        self.__problem = problem
        self.__P = None
        self.iteration = 0
        self.lcon = 0
        self.rcon = 0
        self.eigenvalues = numpy.ndarray((0,), dtype=numpy.float64)
        self.eigenvalue_errors = EstimatedErrors()
        self.eigenvector_errors = EstimatedErrors()
        self.residual_norms = numpy.ndarray((0,), dtype=numpy.float32)
        self.convergence_status = numpy.ndarray((0,), dtype=numpy.int32)

    def set_preconditioner(self, P):
        self.__P = P

    def convergence_data(self, what='residual', which=0):
        if what.find('block') > -1:
            return self.block_size
        elif what.find('res') > -1 and what.find('vec') == -1:
            max_lmd = numpy.amax(abs(self.lmd))
            if self.lcon + self.rcon > 0:
                max_lmd = max(max_lmd, numpy.amax(abs(self.eigenvalues)))
            return self.res[which]/max_lmd
        elif what.find('val') > -1:
            if what.find('err') > -1:
                err = self.err_lmd[:, which]
                if what.find('k'):
                    return err[0]
                else:
                    return err[1]
            else:
                return self.lmd[which]
        elif what.find('vec') > -1:
            err = self.err_X[:, which]
            if what.find('k') > -1:
                return err[0]
            else:
                return err[1]
        else:
            raise ValueError('convergence data %s not found' % what)

    def solve(self, eigenvectors, options=Options(), which=(-1, -1), \
        extra=(-1, -1), init=(None, None)):

        verb = options.verbosity

        left = int(which[0])
        right = int(which[1])
        if left == 0 and right == 0:
            if verb > -1:
                print('No eigenpairs requested, quit')
            return 0

        m = int(options.block_size)
        if m < 0:
            m = _default_block_size(which, extra, init, options.threads)
        else:
            if left == 0 or right == 0:
                if m < 3:
                    if verb > -1:
                        print('Block size %d too small, will use 3 instead' % m)
                    m = 3
            else:
                if m < 4:
                    if verb > -1:
                        print('Block size %d too small, will use 4 instead' % m)
                    m = 4
        self.block_size = m

        n = eigenvectors.dimension()

        #output
        self.iteration = 0
        self.lcon = 0
        self.rcon = 0
        self.eigenvalues = numpy.ndarray((0,), dtype=numpy.float64)
        self.eigenvalue_errors = EstimatedErrors()
        self.eigenvector_errors = EstimatedErrors()
        self.residual_norms = numpy.ndarray((0,), dtype=numpy.float32)
        self.convergence_status = numpy.ndarray((0,), dtype=numpy.int32)

        if m < n//2:
            try:
                status = self._solve(eigenvectors, options, which, extra, init)
                if status > 1:
                    return status
            except _Error as err:
                print('%s' % err.value)
                return -1
        else:
            status = 1

        if status == 0:
            return 0 # success

        Xc = eigenvectors
        nc = Xc.nvec()
        m = n - nc
        if verb > -1:
            print('%d eigenpairs not computed' % m)

        X = eigenvectors.new_vectors(m)
        X.fill_random()
        Y = X.new_vectors(m)
        Z = X.new_vectors(m)
        std = (self.__problem.type() == 's')
        pro = (self.__problem.type() == 'p')

        A = self.__problem.A()
        B = self.__problem.B()
        data_type = eigenvectors.data_type()

        if nc > 0:
            if not std:
                BXc = eigenvectors.clone()
                if nc > 0:
                    B(Xc, BXc)
            else:
                BXc = Xc
            if nc > 0:
                Gc = BXc.dot(Xc)
                # approximate inverse of Gc
                Gci = 2*numpy.identity(nc, dtype=data_type) - Gc
            Q = numpy.dot(Gci, X.dot(BXc))
            X.add(Xc, -1.0, Q)
            Q = numpy.dot(Gci, X.dot(BXc))
            X.add(Xc, -1.0, Q)

        if not std:
            B(X, Y)
            XBX = Y.dot(X)
        else:
            XBX = X.dot(X)
        if pro:
            A(Y, Z)
            XAX = Z.dot(Y)
        else:
            A(X, Z)
            XAX = Z.dot(X)
        lmdx, Q = sla.eigh(XAX, XBX, turbo=False, overwrite_a=True, overwrite_b=True)
        X.multiply(Q, Z)
        Z.copy(X)
        eigenvectors.append(X)
        self.eigenvalues = numpy.concatenate((self.eigenvalues, lmdx))

        return 0

    def _solve(self, eigenvectors, options, which, extra, init):

        verb = options.verbosity

        left = int(which[0])
        right = int(which[1])

        m = self.block_size
        if left == 0:
            r = 0.0
            l = 1
        elif right == 0:
            r = 1.0
            l = m - 1
        elif left > 0 and right > 0:
            r = left/(left + 1.0*right)
            l = int(round(r*m))
            if l < 2:
                l = 2
            if l > m - 2:
                l = m - 2
        else:
            r = 0.5
            l = m//2
        lr_ratio = r
        block_size = m
        left_block_size = l

        extra_left = int(extra[0])
        extra_right = int(extra[1])
        if left >= 0:
            if extra_left > 0:
                left_total = left + extra_left
            else:
                left_total = max(left + 1, left_block_size)
            if verb > 2:
                print('left total: %d' % left_total)
        if right >= 0:
            if extra_right > 0:
                right_total = right + extra_right
            else:
                right_total = max(right + 1, block_size - left_block_size)
            if verb > 2:
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
        data_type = vector.data_type()
        epsilon = numpy.finfo(data_type).eps

        # convergence data
        self.cnv = numpy.zeros((m,), dtype=numpy.int32)
        self.lmd = numpy.zeros((m,), dtype=numpy.float64)
        self.res = -numpy.ones((m,), dtype=numpy.float32)
        self.min_res = -numpy.ones((m,), dtype=numpy.float32)
        self.err_lmd = -numpy.ones((2, m,), dtype=numpy.float32)
        self.err_X = -numpy.ones((2, m,), dtype=numpy.float32)

        # convergence criteria
        if options.convergence_criteria is None:
            convergence_criteria = DefaultConvergenceCriteria()
        else:
            convergence_criteria = options.convergence_criteria

        # data for estimating error in computing residuals
        norm_AX = numpy.zeros((m,), dtype=numpy.float32)

        # convergence history data
        iterations = numpy.zeros((m,), dtype=numpy.int32)
        dlmd = numpy.zeros((m, RECORDS), dtype=numpy.float32)
        dX = numpy.ones((m,), dtype=numpy.float32)
        acf = numpy.ones((2, m,), dtype=numpy.float32)

        # workspace
        X = vector.new_vectors(m)
        X.fill_random()
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
        AZ = AY
        BZ = BY

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
        detect_stagn = options.detect_stagnation
        lmd = self.lmd
        res = self.res
        min_res = self.min_res
        err_lmd = self.err_lmd
        err_X = self.err_X
        A = self.__problem.A()
        B = self.__problem.B()
        P = self.__P

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
            Gci = 2*numpy.identity(nc, dtype=data_type) - Gc

        # initialize
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
            X.add(Xc, -1.0, Q)

        if not std:
            B(X, BX)
        XBX = BX.dot(X)

        # do pivoted Cholesky for XBX to eliminate linear dependent X
        U = XBX.copy()
        ind, dropped = _piv_chol(U, 0, 1e-2)
        if dropped > 0:
            if verb > 0:
                print('dropped %d initial vectors out of %d' % (dropped, nx))
            # drop linear dependent initial vectors
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
                Xc.multiply(Q, W)
                X.add(W, -1.0)
                if not std:
                    BXc.multiply(Q, W)
                    BX.add(W, -1.0)
            nx = m
            X.select(nx)
            if not std:
                BX.select(nx)
            XBX = BX.dot(X)

        # Rayleigh-Ritz in the initial space
        if pro:
            A(BX, AX)
            XAX = AX.dot(BX)
        else:
            A(X, AX)
            XAX = AX.dot(X)
        lmdx, Q = sla.eigh(XAX, XBX, turbo=False)
        W.select(m)
        X.multiply(Q, W)
        W.copy(X)
        AX.multiply(Q, W)
        W.copy(AX)
        if not std:
            BX.multiply(Q, Z)
            Z.copy(BX)

# ===== main CG loop
        max_iter = options.max_iter
        min_iter = options.min_iter
        if max_iter < 0:
            max_iter = 100
        self.iteration = 0
        while True:

            maxit = 0
            if left != 0:
                maxit = numpy.amax(iterations[:left_block_size])
            if right != 0:
                maxit = max(maxit, numpy.amax(iterations[left_block_size:]))
            if maxit > max_iter:
                break

            if verb > 0:
                print('------------- iteration %d' % self.iteration)

            if pro:
                XAX = AX.dot(BX)
            else:
                XAX = AX.dot(X)
            XBX = BX.dot(X)
            da = XAX.diagonal()
            db = XBX.diagonal()
            new_lmd = _real(da/db)

            # estimate error in residual computation due to the error in
            # computing AX, to be used in detecting convergence stagnation
            Lmd = numpy.zeros((nx, nx))
            Lmd[range(nx), range(nx)] = lmdx #new_lmd
            RX = XAX - numpy.dot(XBX, Lmd)
            delta_R = nla.norm(RX, axis=0)
            if gen:
                s = numpy.sqrt(abs(X.dots(X)))
                delta_R /= s
            rv_err = numpy.amax(abs(new_lmd - lmdx))
            if verb > 2:
                print('Ritz values error: %.1e' % rv_err)
                print('Ritz vectors non-orthonormality: %.1e' % \
                      numpy.amax(abs(XBX - numpy.eye(nx))))

            if self.iteration > 0:
                # compute eigenvalue decrements
                for i in range(nx):
                    if iterations[ix + i]:
                        delta = lmd[ix + i] - new_lmd[i]
                        eps = math.sqrt(epsilon)
                        eps *= max(abs(lmd[ix + i]), abs(new_lmd[i]))
                        if abs(delta) > eps:
                            dlmd[ix + i, rec - 1] = delta
                    iterations[ix + i] += 1
                if verb > 3:
                    print('eigenvalues shifts history:')
                    print(numpy.array_str(dlmd[ix : ix + nx, :rec].T, \
                                          precision=2))

            lmd[ix : ix + nx] = new_lmd

            # compute residuals
            # std: A X - X lmd
            # gen: A X - B X lmd
            # pro: A B X - X lmd
            W.select(nx, ix)
            Y.select(nx)
            AX.copy(W)
            norm_AX[ix : ix + nx] = numpy.sqrt(abs(W.dots(W)))
            if gen:
                W.add(BX, -lmd[ix : ix + nx])
            else:
                W.add(X, -lmd[ix : ix + nx])

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
                    W.add(BXc, -1.0, Q)
                else:
                    W.add(Xc, -1.0, Q)

            if pro:
                W.copy(Y)
                B(Y, W)
                s = W.dots(Y)
            else:
                s = W.dots(W)
            res[ix : ix + nx] = numpy.sqrt(abs(s))
            if self.iteration == 0:
                min_res[:] = res*epsilon

            # kinematic error estimates
            if rec > 3: # sufficient history available
                for i in range(nx):
                    if dX[ix + i] > 0.01:
                        err_X[0, ix + i] = -1.0
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
                    if qi >= 1.0:
                        continue
                    # esimate error based on a.c.f.
                    theta = qi/(1 - qi)
                    d = theta*dlmd[ix + i, rec - 1]
                    err_lmd[0, ix + i] = d
                    qx = math.sqrt(qi)
                    err_X[0, ix + i] = dX[ix + i]*qx/(1 - qx)

            if not gen:
                # residual-based error estimates:
                # asymptotic Lehmann for eigenvalues
                # generalized (extended gap) Davis-Kahan for eigenvectors;
                # not valid for the generalized eigenvalue problem
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
                    if verb > 2:
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
                msg = 'eigenvalue     residual  ' + \
                'estimated errors (kinematic/residual)' + \
                '   a.c.f.'
                print(msg)
                for i in range(block_size):
                    print('%14e %8.1e  %.1e / %.1e    %.1e / %.1e  %.3e  %d' % \
                          (lmd[i], res[i], \
                          abs(err_lmd[0, i]), abs(err_lmd[1, i]), \
                          abs(err_X[0, i]), abs(err_X[1, i]), \
                          acf[0, i], self.cnv[i]))

            lcon = 0
            for i in range(leftX - max(1, leftX//4)):
                j = self.lcon + i
                k = ix + i
                it = iterations[k]
                if it < min_iter:
                    break
                res_err = min_res[k] + 10*delta_R[i]
                dlmd1 = abs(dlmd[k, rec - 1])
                dlmd2 = abs(dlmd[k, rec - 2])
                if convergence_criteria.satisfied(self, k):
                    if verb > 0:
                        msg = 'left eigenpair %d converged' + \
                        ' after %d iterations,\n' + \
                        ' eigenvalue %e, error %.1e / %.1e'
                        it = iterations[k]
                        print(msg % (j, it, lmd[k], err_X[0, k], err_X[1, k]))
                    lcon += 1
                    self.cnv[k] = self.iteration + 1
                elif detect_stagn and res[k] >= 0 and \
                    res[k] < res_err and dlmd1 > dlmd2:
                    if verb > 0:
                        msg = 'left eigenpair %d stagnated,\n' + \
                        ' eigenvalue %e, error %.1e / %.1e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    lcon += 1
                    self.cnv[k] = -self.iteration - 1
                else:
                    break

            rcon = 0
            for i in range(rightX - max(1, rightX//4)):
                j = self.rcon + i
                k = ix + nx - i - 1
                it = iterations[k]
                if it < min_iter:
                    break
                res_err = min_res[k] + 10*delta_R[nx - i - 1]
                dlmd1 = abs(dlmd[k, rec - 1])
                dlmd2 = abs(dlmd[k, rec - 2])
                if convergence_criteria.satisfied(self, k):
                    if verb > 0:
                        msg = 'right eigenpair %d converged' + \
                        ' after %d iterations,\n' + \
                        ' eigenvalue %e, residual %.1e, error %.1e / %.1e'
                        print(msg % (j, it, lmd[k], res[k], err_X[0, k], err_X[1, k]))
                    rcon += 1
                    self.cnv[k] = self.iteration + 1
                elif detect_stagn and res[k] >= 0 and \
                    res[k] < res_err and dlmd1 > dlmd2:
                    if verb > 0:
                        msg = 'right eigenpair %d stagnated,\n' + \
                        ' eigenvalue %e, error %.1e / %.1e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    rcon += 1
                    self.cnv[k] = -self.iteration - 1
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
                self.convergence_status = numpy.concatenate \
                    ((self.convergence_status, self.cnv[ix : ix + lcon]))
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
                    Gc = numpy.concatenate((Gc, Gu), axis=1)
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
                self.convergence_status = numpy.concatenate \
                    ((self.convergence_status, self.cnv[jx - rcon : jx]))
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
                    Gc = numpy.concatenate((Gc, Gu), axis=1)
                    Gc = numpy.concatenate((Gc, Gl))
                ncon += rcon
            if ncon > 0:
                H = Gc - numpy.identity(ncon, dtype=data_type)
                if verb > 2:
                    print('Gram error: %e' % nla.norm(H))
                # approximate inverse, good enough for Gram error up to 1e-8
                Gci = 2*numpy.identity(ncon, dtype=data_type) - Gc

            self.lcon += lcon
            self.rcon += rcon
            if options.stopping_criteria is not None:
                if options.stopping_criteria.satisfied(self):
                    return 0
            left_converged = left >= 0 and self.lcon >= left
            right_converged = right >= 0 and self.rcon >= right
            if left_converged and right_converged:
                return 0
            lim = options.max_quota * eigenvectors.dimension()
            if eigenvectors.nvec() > lim:
                return 1

            leftX -= lcon
            rightX -= rcon

            # re-select Xs, AXs, BXs accordingly
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

            if not pro:
                if P is None:
                    W.copy(Y)
                else:
                    P(W, Y)

            if nz > 0:
                # compute the conjugation matrix
                if pro:
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
                sy = numpy.sqrt(abs(Y.dots(Y)))
                sz = numpy.sqrt(abs(Z.dots(Z)))
                Beta = numpy.ndarray((nz, ny), dtype=data_type)
                for iz in range(nz):
                    for iy in range(ny):
                        s = sy[iy]/sz[iz]
                        if abs(Num[iz, iy]) >= 100*s*abs(Den[iz, iy]):
                            Beta[iz, iy] = 0.0
                        else:
                            Beta[iz, iy] = Num[iz, iy]/Den[iz, iy]

                # conjugate search directions
                AZ.select(ny)
                Y.add(Z, -1.0, Beta)
                if pro: # if gen or nz == 0, BY computed later
                    W.add(BZ, -1.0, Beta)
                    BY.select(ny)
                    W.copy(BY)
            elif pro:
                BY.select(ny)
                W.copy(BY)

            Q = Y.dot(BX)
            Y.add(X, -1.0, Q)
            if pro:
                BY.add(BX, -1.0, Q)

            if Xc.nvec() > 0:
                # orthogonalize Y to Xc
                # std: W := W - Xc Xc* W (not needed if P is None)
                # gen: W := W - Xc BXc* W
                # pro: W := W - Xc BXc* W (not needed if P is None)
                Q = numpy.dot(Gci, Y.dot(BXc))
                Y.add(Xc, -1.0, Q)
                if pro:
                    BY.add(BXc, -1.0, Q)

            # compute (B-)Gram matrix for (X,Y)
            if std:
                s = numpy.sqrt(abs(Y.dots(Y)))
                Y.scale(s)
                if nx > 0:
                    XBY = Y.dot(X)
                YBY = Y.dot(Y)
            else:
                BY.select(Y.nvec())
                if not pro: # or nz == 0:
                    B(Y, BY)
                s = numpy.sqrt(abs(BY.dots(Y)))
                Y.scale(s)
                BY.scale(s)
                if nx > 0:
                    XBY = BY.dot(X)
                YBY = BY.dot(Y)

            if nx > 0:
                YBX = _conjugate(XBY)
                GB = numpy.concatenate((XBX, YBX))
                H = numpy.concatenate((XBY, YBY))
                GB = numpy.concatenate((GB, H), axis=1)
            else:
                GB = YBY

            # do pivoted Cholesky for GB
            U = GB
            ny = Y.nvec()
            if data_type == numpy.float32 or data_type == numpy.complex64:
                eps = 1e-4
            else:
                eps = 1e-8
            ind, dropped = _piv_chol(U, nx, eps)
            if dropped > 0:
                if verb > 0:
                    print('dropped %d search directions out of %d' \
                          % (dropped, ny))

            ny -= dropped
            if ny < 1:
                if verb > -1:
                    print('no search directions left, terminating')
                return 3

            # re-arrange/drop-linear-dependent search directions
            nxy = nx + ny
            U = U[:nxy, :nxy]

            indy = ind[nx: nxy]
            for i in range(ny):
                indy[i] -= nx
            W.select(ny)
            Y.copy(W, indy[:ny])
            Y.select(ny)
            W.copy(Y)
            AY.select(ny)
            if not std:
                BY.copy(W, indy[:ny])
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
                YAX = _conjugate(XAY)
                GA = numpy.concatenate((XAX, YAX))
                H = numpy.concatenate((XAY, YAY))
                GA = numpy.concatenate((GA, H), axis=1)
            else:
                GA = YAY

            # solve Rayleigh-Ritz eigenproblem
            G = _transform(GA, U)
            YAY = G[nx : nxy, nx : nxy]
            lmdy, Qy = sla.eigh(YAY)
            G[:, nx : nxy] = numpy.dot(G[:, nx : nxy], Qy)
            if nx > 0:
                G[nx : nxy, :nx] = _conjugate(G[:nx, nx : nxy])
            G[nx : nxy, nx : nxy] = numpy.dot(_conjugate(Qy), G[nx : nxy, nx : nxy])

            if G.dtype.kind == 'c':
                G = G.astype(numpy.complex128)
            else:
                G = G.astype(numpy.float64)

            lmdxy, Q = sla.eigh(G)

            lmdxy = lmdxy.astype(lmdy.dtype)
            Q = Q.astype(Qy.dtype)

            # estimate changes in eigenvalues and eigenvectors
            lmdx = numpy.concatenate \
                ((lmdxy[:leftX], lmdxy[nxy - rightX:]))
            lmdy = lmdxy[leftX : nxy - rightX]
            QX = numpy.concatenate \
                ((Q[:, :leftX], Q[:, nxy - rightX:]), axis=1)
            QYX = QX[nx:, :].copy()
            lmdX = numpy.ndarray((1, nx))
            lmdY = numpy.ndarray((ny, 1))
            lmdX[0, :] = lmdx
            lmdY[:, 0] = lmdy
            Delta = (lmdY - lmdX)*QYX*QYX
            dX[ix : ix + nx] = nla.norm(QYX, axis=0)
            if rec == RECORDS:
                for i in range(rec - 1):
                    dlmd[:, i] = dlmd[:, i + 1]
            else:
                rec += 1
            dlmd[ix : ix + nx, rec - 1] = _real(numpy.sum(Delta, axis=0))

            # select new numbers of left and right eigenpairs
            if left < 0:
                shift_left = ix
            elif lcon > 0:
                shift_left = max(0, left_total - self.lcon - leftX)
                shift_left = min(shift_left, ix)
            else:
                shift_left = 0
            if right < 0:
                shift_right = block_size - ix - nx
            elif rcon > 0:
                shift_right = max(0, right_total - self.rcon - rightX)
                shift_right = min(shift_right, block_size - ix - nx)
            else:
                shift_right = 0
            if shift_left + shift_right > ny:
                shift_left = min(shift_left, int(round(lr_ratio*ny)))
                shift_right = min(shift_right, ny - shift_left)
            if left > 0 and lcon > 0 and self.lcon >= left: # left side converged
                if verb > 0:
                    print('left side converged')
                leftX_new = 0
                l = left_block_size
                rightX_new = min(nxy, l + rightX + shift_right)
                left_block_size_new = l + rightX + shift_right - rightX_new
                shift_left = -leftX - lcon
                lr_ratio = 0.0
                ix_new = left_block_size_new
            elif right > 0 and rcon > 0 and self.rcon >= right: # right side converged
                if verb > 0:
                    print('right side converged')
                ix_new = ix - shift_left
                leftX_new = min(nxy, block_size - ix_new)
                rightX_new = 0
                shift_right = -rightX - rcon
                left_block_size_new = ix + leftX + rightX
                lr_ratio = 1.0
            else:
                leftX_new = leftX + shift_left
                rightX_new = rightX + shift_right
                left_block_size_new = left_block_size
                ix_new = ix - shift_left
            nx_new = leftX_new + rightX_new
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
                    j = i + shift_left
                    cnv[i] = cnv[j]
                    lmd[i] = lmd[j]
                    res[i] = res[j]
                    acf[:, i] = acf[:, j]
                    err_lmd[:, i] = err_lmd[:, j]
                    dlmd[i, :] = dlmd[j, :]
                    err_X[:, i] = err_X[:, j]
                    dX[i] = dX[j]
                    iterations[i] = iterations[j]
            if shift_left >= 0:
                for i in range(l - shift_left, nl):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)
            else:
                for i in range(l):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)
            if shift_right > 0:
                for i in range(m - 1, l + shift_right - 1, -1):
                    j = i - shift_right
                    cnv[i] = cnv[j]
                    lmd[i] = lmd[j]
                    res[i] = res[j]
                    acf[:, i] = acf[:, j]
                    err_lmd[:, i] = err_lmd[:, j]
                    dlmd[i, :] = dlmd[j, :]
                    err_X[:, i] = err_X[:, j]
                    dX[i] = dX[j]
                    iterations[i] = iterations[j]
            if shift_right >= 0:
                for i in range(l + shift_right - 1, nl - 1, -1):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)
            else:
                for i in range(l, block_size):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)

            # compute RR coefficients for X and 'old search directions' Z
            # by re-arranging columns of Q
            Q[nx : nxy, :] = numpy.dot(Qy, Q[nx : nxy, :])
            Q = sla.solve_triangular(U, Q)
            lmdx = numpy.concatenate \
                ((lmdxy[:leftX_new], lmdxy[nxy - rightX_new:]))
            QX = numpy.concatenate \
                ((Q[:, :leftX_new], Q[:, nxy - rightX_new:]), axis=1)
            lft = leftX_new
            rgt = rightX_new
            nz = nxy - lft - rgt
            lmdz = lmdxy[lft : nxy - rgt]
            QZ = Q[:, lft : nxy - rgt]
            if nx > 0:
                QXX = QX[:nx, :].copy()
            QYX = QX[nx:, :].copy()
            if nx > 0:
                QXZ = QZ[:nx, :].copy()
            QYZ = QZ[nx:, :].copy()

            # update X and 'old search directions' Z and their A- and B-images
            W.select(nx_new)
            Z.select(nx_new)
            if nx > 0:
                AX.multiply(QXX, W)
                W.add(AY, 1.0, QYX)
            else:
                AY.multiply(QYX, W)
            if nz > 0:
                Z.select(nz)
                AY.multiply(QYZ, Z)
                AZ.select(nz)
                if nx > 0:
                    Z.add(AX, 1.0, QXZ)
                Z.copy(AZ)
            AX.select(nx_new, ix_new)
            W.copy(AX)
            if not std:
                Z.select(nx_new)
                if nx > 0:
                    BX.multiply(QXX, W)
                    W.add(BY, 1.0, QYX)
                else:
                    BY.multiply(QYX, W)
                if nz > 0:
                    Z.select(nz)
                    BY.multiply(QYZ, Z)
                    BZ.select(nz)
                    if nx > 0:
                        Z.add(BX, 1.0, QXZ)
                    Z.copy(BZ)
                BX.select(nx_new, ix_new)
                W.copy(BX)
            else:
                BZ = Z
            Z.select(nx_new)
            if nx > 0:
                X.multiply(QXX, W)
                W.add(Y, 1.0, QYX)
            else:
                Y.multiply(QYX, W)
            if nz > 0:
                Z.select(nz)
                Y.multiply(QYZ, Z)
                if nx > 0:
                    Z.add(X, 1.0, QXZ)
            X.select(nx_new, ix_new)
            W.copy(X)

            nx = nx_new
            ix = ix_new
            leftX = leftX_new
            rightX = rightX_new
            left_block_size = left_block_size_new
            self.iteration += 1

        return 2

class _Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return '??? ' + repr(self.value)

def _conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj().T
    else:
        return a.T

def _real(a):
    if a.dtype.kind == 'c':
        return a.real
    else:
        return a

def _transform(A, U):
    B = sla.solve_triangular(_conjugate(U), _conjugate(A), lower=True)
    A = sla.solve_triangular(_conjugate(U), _conjugate(B), lower=True)
    return A

def _default_block_size(which, extra, init, threads):
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
    if threads <= 8:
        threads = 8
    if left == 0 and right == 0:
        return 0
    if left <= 0 and right <= 0:
        if init_left == 0 and init_right == 0:
            if left < 0 and right < 0:
                return 2*threads
            else:
                return threads
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

def _reset_cnv_data(i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations):
    cnv[i] = 0
    res[i] = -1.0
    acf[:, i] = 1.0
    err_lmd[:, i] = -1.0
    dlmd[i, :] = 0
    err_X[:, i] = -1.0
    dX[i] = 1.0
    iterations[i] = 0

def _piv_chol(A, k, eps, blk=64, verb=0):
    n = A.shape[0]
    buff = A[0, :].copy()
    ind = [i for i in range(n)]
    drop_case = 0
    dropped = 0
    last_check = -1
    if k > 0:
        U = sla.cholesky(A[:k, :k])
        A[:k, :k] = U.copy()
        A[:k, k : n] = sla.solve_triangular \
                       (_conjugate(U), A[:k, k : n], lower=True)
        A[k : n, :k].fill(0.0)
        A[k : n, k : n] -= numpy.dot(_conjugate(A[:k, k : n]), A[:k, k : n])
    l = k
    for i in range(k, n):
        s = numpy.diag(A[i : n, i : n]).copy()
        if i > l:
            t = nla.norm(A[l : i, i : n], axis=0)
            s -= t*t
        j = i + numpy.argmax(s)
        if i != j:
            buff[:] = A[i, :]
            A[i, :] = A[j, :]
            A[j, :] = buff
            buff[:] = A[:, i]
            A[:, i] = A[:, j]
            A[:, j] = buff
            ind[i], ind[j] = ind[j], ind[i]
        if i > l:
            A[i, i : n] -= numpy.dot(_conjugate(A[l : i, i]), A[l : i, i : n])
        last_piv = A[i, i].real
        if last_piv <= eps:
            A[i : n, :].fill(0.0)
            drop_case = 1
            dropped = n - i
            break
        A[i, i] = math.sqrt(last_piv)
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
        if i - l == blk - 1 or i == n - 1:
            last_check = i
            lmin = _estimate_lmin(A[: i + 1, : i + 1])
            lmax = _estimate_lmax(A[: i + 1, : i + 1])
            if verb > 0:
                print('%e %e %e' % (A[i, i], lmin, lmax))
            if lmin/lmax <= eps:
                A[i : n, :].fill(0.0)
                drop_case = 2
                dropped = n - i
                break
        if i - l == blk - 1 and i < n - 1:
            j = i + 1
            A[j : n, j : n] -= numpy.dot(_conjugate(A[l : j, j : n]), \
                A[l : j, j : n])
            l += blk
    if last_check < n - 1 and drop_case != 2:
        i = last_check
        j = n - dropped - 1
        while i < j:
            m = i + (j - i + 1)//2
            lmin = _estimate_lmin(A[: m + 1, : m + 1])
            lmax = _estimate_lmax(A[: m + 1, : m + 1])
            if verb > 0:
                print('%d %e %e' % (m, lmin, lmax))
            if lmin/lmax <= eps:
                if j > m:
                    j = m
                    continue
                else:
                    A[j : n, :].fill(0.0)
                    dropped = n - j
                    last_piv = A[j - 1, j - 1]**2
                    break
            else:
                i = m
                continue
    return ind, dropped

def _estimate_lmax(U):
    U = numpy.triu(U)
    return sla.norm(numpy.dot(_conjugate(U), U), ord=1)
def _estimate_lmin(U):
    n = U.shape[0]
    if U.dtype.kind == 'c':
        tr = 2
    else:
        tr = 1
    x = numpy.ones((n,), dtype=U.dtype)
    s = numpy.dot(x, x)
    for i in range(3):
        y = sla.solve_triangular(U, x, trans=tr)
        t = numpy.dot(y, y)
        rq = s/t
        x = sla.solve_triangular(U, y)
        s = numpy.dot(x, x)
    return rq
