# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:37:46 2017

@author: wps46139
"""

#        solver.rr[k, j : j + ny, i : i + nx] = \
#            alpha*numpy.dot( \
#                w[ky, :, jy : jy + ny].transpose(), 
#                w[kx, :, jx : jx + nx]) + \
#            beta*solver.rr[k, j : j + ny, i : i + nx]

#        if job == TRANSFORM:
#            w[ky, :, jy : jy + ny] = \
#                numpy.dot(w[kx, :, jx : jx + nx], \
#                    solver.rr[k, j : j + ny, i : i + nx].transpose())
#            w[kx, :, jx : jx + ny] = w[ky, :, jy : jy + ny]
#        else:
#            w[ky, :, jy : jy + ny] = \
#                alpha*numpy.dot(w[kx, :, jx : jx + nx], \
#                    solver.rr[k, j : j + ny, i : i + nx].transpose()) + \
#                beta*w[ky, :, jy : jy + ny]

#        print('XAX:')
#        print(XAX)
#        print('XBX:')
#        print(XBX)

#        print('q:')
#        print(q)

 #       W = vector.new_vectors(m)

        #Y = vector.new_vectors(nx)

#        if W.nvec() != ny:
#            del W
#            W = vector.new_vectors(ny)

#        if Y.nvec() != ny:
#            del Y
#            Y = vector.new_vectors(ny)

#            if BY.nvec() != ny:
#                del BY
#                BY = vector.new_vectors(ny)

#        AY = vector.new_vectors(ny)

#        if W.nvec() != nx_new:
#            del W
#            W = vector.new_vectors(nx_new)
        #print(W.data().flags['C_CONTIGUOUS'])
#        Z = vector.new_vectors(nx_new)

#            if Z.nvec() != nz:
#                del Z
#                Z = vector.new_vectors(nz)

#            del AY
#            AZ = vector.new_vectors(nz)

#            if Z.nvec() != nx_new:
#                del Z
#                Z = vector.new_vectors(nx_new)

#            if Z.nvec() != nz:
#                del Z
#                Z = vector.new_vectors(nz)

#            del BY
#            BZ = vector.new_vectors(nz)

#        blocks = 4
#        if self.__min_opA:
#            self.__AX = vector.new_vectors(m)
#            self.__AY = vector.new_vectors(m)
#            blocks += 2
#        else:
#            self.__AX = None
#            self.__AY = None
##            self.__AX = self.__W
##            #self.__AYZ = self.__W
#        if self.__min_opB or problem.type() == 'p':
#            self.__BX = vector.new_vectors(m)
#            self.__BY = vector.new_vectors(m)
#            blocks += 2
#        else:
#            self.__BX = None
#            self.__BY = None
##            if problem.type() == 's':
##                self.__BX = self.__X
##            else:
##                self.__BX = self.__Y
##            #self.__BYZ = self.__Z                
#        self.__blocks = blocks

#    if isinstance(A[0,0], complex):
#        B = sla.solve_triangular(U.conj().T, A.conj().T, lower = True)
#        A = sla.solve_triangular(U.conj().T, B.conj().T, lower = True)
#    else:
#        B = sla.solve_triangular(U.T, A.T, lower = True)
#        A = sla.solve_triangular(U.T, B.T, lower = True)

#        if isinstance(arg, (numbers.Number, numpy.ndarray)):
#            self.__vector = NDArrayVectors(arg)
#        else:
#            self.__vector = arg

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

            print(QXX.flags['C_CONTIGUOUS'])
            print(QYX.flags['C_CONTIGUOUS'])
            print(QXZ.flags['C_CONTIGUOUS'])
            print(QYZ.flags['C_CONTIGUOUS'])
#            QX = numpy.concatenate((QXX, QYX))
#            G = numpy.dot(QX.T, numpy.dot(GB, QX))
#            print(G.diagonal())
#            #I = numpy.dot(QX.T, QX)
#            I = BX.dot(X)
#            print(I.diagonal())

#            Q = numpy.concatenate((QX, QZ), axis = 1)

##        A(X, AX)
#        if pro:
#            XAX = AX.dot(BX)
#        else:
#            XAX = AX.dot(X)
#        XBX = BX.dot(X)
#        da = XAX.diagonal()
#        db = XBX.diagonal()
##        print('diag(XAX):')
##        print(da)
##        print('diag(XBX):')
##        print(db)
#        lmd = da/db
#        print('lmd:')
#        print(lmd)
#        AX.copy(W)
#        if gen:
#            W.add(BX, -lmd)
#        else:
#            W.add(X, -lmd)
#        s = W.dots(W)
#        print('residual norms:')
#        print(numpy.sqrt(s))

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

            #print(Xc.dimension(), Xc.nvec())
            #print(BXc.dimension(), BXc.nvec())

##    def set_type(self, problem_type):
##        t = problem_type[0]
##        if t == 's' and B is not None:
##            print('WARNING: B will be ignored')
##        elif t != 's' and B is None:
##            print('WARNING: no B defined, standard problem will be solved')
##        if t != 's' and t != 'g' and t != 'p':
##            print('WARNING: unknown problem type, assumed standard')
##        self.__type = t

#        self.__min_opA = options.min_opA
#        self.__min_opB = options.min_opB and problem.type() != 's'

            #print('residual norms:')
            #print(res)

#v = NDArrayVectors(numpy.ndarray((0, n)))

    #        print('Gram matrix for (X,Y):')
    #        print(GB)

#            lmdxy, Q = sla.eigh(GA)
#            print(lmdxy)
#            Q = sla.solve_triangular(U, Q)

#            XAX = G[:nx, :nx]
#            XAY = G[:nx, nx : nxy]
#            YAX = G[nx : nxy, :nx]

#            for i in range(nx):
#                s = 0.0
#                for j in range(ny):
#                    t = abs(XAY[i, j])
#                    s += t*t/abs(lmdy[j] - lmd[ix + i])
#                dlmd[ix + i] = s
            for i in range(nx):
                s = 0.0
                for j in range(ny):
                    t = abs(G[i, nx + j])/abs(lmdy[j] - lmd[ix + i])
                    s += t*t
                dX[ix + i] = math.sqrt(s)

#                    q[m + ix + i] = qi # rough a.c.f. estimate
#                    if qi > 0.9999999: # too large, skip refinement
#                        continue

#                    err_lmd[m + ix + i] = d
#                    qi = (d/(s + d))**(1.0/k)
#                    q[ix + i] = qi # refined a.c.f. estimate
#                    theta = qi/(1 - qi)
#                    d = theta*dlmd[ix + i, rec - 1]
#                    err_lmd[ix + i] = d

#            print(X.selected())
#            print(AX.selected())
#            print(BX.selected())
#            B(X, BX)
#            A(BX, AX)

#            print(XAX)
#            print(XBX)

            W.select(ny)
            print(Y.selected())
            print(W.selected())
            B(Y, W)
            W.add(BY, -1.0)
            s = W.dots(W)
            print('BY err: ', s)

            W.select(ny)
            B(Y, W)
            W.add(BY, -1.0)
            s = W.dots(W)
            print('BY err: ', s)

            W.select(nx)
            B(X, W)
            W.add(BX, -1.0)
            s = W.dots(W)
            print('BX err: ', s)
            W.select(ny)
            B(Y, W)
            W.add(BY, -1.0)
            s = W.dots(W)
            print('BY err: ', s)

            W.select(nx_new)
            A(BX, W)
            W.add(AX, -1.0)
            s = W.dots(W)
            print('A err: ', s)
            W.select(nx_new)
            B(X, W)
            W.add(BX, -1.0)
            s = W.dots(W)
            print('B err: ', s)

#                elif gen:
#                    BY.select(ny)
#                    B(Y, BY)
#            elif not std:
#                ny = Y.nvec()
#                BY.select(ny)
#                B(Y, BY)

#                    print(dlmd[ix + i, rec - 1], lmd[ix + i] - new_lmd[i], dX[ix + i])

#        extra_left = extra[0]
#        extra_right = extra[1]
#        if extra_left < 0:
#            extra_left = 0 # TODO: set proper default
#        if extra_right < 0:
#            extra_right = 0 # TODO: set proper default
#        left_total = left + extra_left
#        right_total = right + extra_right

#            if left < 0 or right < 0:
#                m = 16
#            else:
#                m = max(2, left_total + right_total)

def block_size(left, right):
    threads = 8
    if left == 0 and right == 0:
        return 0
    if left <= 0 and right <= 0:
        return 2*threads
    left_total = 0
    right_total = 0
    if left > 0:
        left_total = int(round(left*1.2))
    if right > 0:
        right_total = int(round(right*1.2))
    if left < 0:
        left_total = right_total
    if right < 0:
        right_total = left_total
    m = int(left_total + right_total)
    m = 8*((m - 1)//8 + 1)
    if left < 0 or right < 0:
        m = max(m, 2*threads)
    return m

#while True:
#    left = int(input('left: '))
#    if left == 999:
#        break
#    right = int(input('right: '))
#    m = block_size(left, right)
#    print('block size: %d' % m)

#                print(ZAY.shape, ZBY.shape, ny, numpy.diag(lmd[iy : iy + ny]).shape)

#            # estimate eigenvalue and eigenvector shifts
#            if nx > 0:
#                Num = G[:nx, nx : nxy]
#                Num = numpy.absolute(Num)
#                Lmd = numpy.ndarray((1, ny))
#                Mu = numpy.ndarray((nx, 1))
#                Lmd[0, :] = lmdy
#                Mu[:, 0] = lmd[ix : ix + nx]
#                Den = Mu - Lmd
#                # safeguard against division overflow
#                eps = 1e-16
#                exclude = numpy.absolute(Den) < eps*Num
#                Den[exclude] = eps*Num[exclude]
#                dX[ix : ix + nx] = nla.norm(Num/Den, axis = 1)
#                Num = Num*Num
#                if rec == RECORDS:
#                    for i in range(rec - 1):
#                        dlmd[:, i] = dlmd[:, i + 1]
#                else:
#                    rec += 1
#                dlmd[ix : ix + nx, rec - 1] = -numpy.sum(Num/Den, axis = 1)

#                    dlmd[i, :] = dlmd[i + shift_left, :]
#                    dX[i] = dX[i + shift_left]

#                    dlmd[i, :] = 0
#                    dX[i] = 0

#                    dlmd[i, :] = dlmd[i - shift_right, :]
#                    dX[i] = dX[i - shift_right]

#                    dlmd[i, :] = 0
#                    dX[i] = 0

#            print(dX[ix : ix + nx_new])

#            print(dlmd[ix : ix + nx_new, rec - 1])

#def opA(x, y):
#    n = x.dimension()
#    for i in range(n):
#        y.data()[:,i] = (i + 1)*x.data()[:,i]

#def opB(x, y):
#    d = 2*numpy.ones((1, x.dimension()))
#    y.data()[:,:] = d*x.data()
##    y.data()[:,:] = 2*x.data()[:,:]

#def opP(x, y):
#    n = x.dimension()
#    for i in range(n):
#        y.data()[:,i] = x.data()[:,i]/(i + 1)

#    def data_type(self):
#        return self.__data.dtype

#    def axpy(self, q, output):
#        f, n = output.__selected;
#        output.__data[:, f : f + n] += numpy.dot(self.data(), q)

#            if not std:
#                Q = X.dot(BXc)
#            else:
#                Q = X.dot(Xc)

#        else:
#            BX = X

#                    Gc = numpy.concatenate((Gc, Gx[:, :ncon]))
#                    Gc = numpy.concatenate((Gc, conjugate(Gx)), axis = 1)

#                    Gc = numpy.concatenate((Gc, Gx[:, :ncon]))
#                    Gc = numpy.concatenate((Gc, conjugate(Gx)), axis = 1)
                #Gc = (Gc + conjugate(Gc))/2

#            if verb > 1:
#                print('eigenvalue   residual estimated errors')
#                for i in range(block_size):
#                    print('%e %.1e  %.1e %.1e' % \
#                          (lmd[i], res[i], err_lmd[i + m], err_X[i + m]))

#                print(Gc)
#                H = numpy.dot(Gci, Gc)
#                print(H)

                #print(shift_left, shift_left_max)
                #print(shift_right, shift_right_max)

                #print(ny, shift_left, nxy)
                #print('shifts:', shift_left, shift_right)

#                left_block_size_new = block_size - rightX_new

#                left_block_size_new = block_size

#            lmdY = numpy.ndarray((nxy - nx_new, 1))

#            print(QYX.shape)
#            print(nla.norm(QYX, axis = 0).shape)
#            print(dX[ix : ix + nx_new].shape)
#            print(nx_new)

#            print(dX[ix : ix + nx_new])

#            print(dlmd[ix : ix + nx_new, rec - 1])

#            lmdz = lmdxy[leftX_new : nxy - rightX_new]
#            QZ = Q[:, leftX_new : nxy - rightX_new]

            #nz = min(block_size, nxy - nx_new)

            #print(ix, leftX, rightX, left_block_size)

def sort_eigenpairs(lmd, u, err_lmd, err_X):
    ind = numpy.argsort(lmd)
    w = u.new_vectors(u.nvec())
    lmd = lmd[ind]
    err_lmd = err_lmd[ind]
    err_X = err_X[ind]
    u.copy(w, ind)
    w.copy(u)
    return lmd, u, err_lmd, err_X

#    lmdu, u, err_lmd, err_X = sort_eigenpairs\
#        (lmdu, u, solver.errors_val, solver.errors_vec)

##    q = u.dot(u)
##    print(q)

#    lmdv, v, err_lmd, err_X = sort_eigenpairs\
#        (lmdv, v, solver.errors_val, solver.errors_vec)

#    for i in range(lcon + rcon):
#        print('%e %e' % (t[i], s[i]))
#    return s, t

    s = numpy.concatenate((sl, sr))

#        self.errors_val = numpy.ndarray((0,), dtype = numpy.float32)
#        self.errors_vec = numpy.ndarray((0,), dtype = numpy.float32)

#        self.err_lmd = -numpy.ones((mm,), dtype = numpy.float32)
#        self.err_X = -numpy.ones((mm,), dtype = numpy.float32)

#        acf = numpy.ones((mm,), dtype = numpy.float32)

#                    err_lmd[i + m] = s*s/(t - lmd[i])
#                    err_X[i + m] = s/(t - lmd[i])

#                    err_lmd[i + m] = s*s/(lmd[i] - t)
#                    err_X[i + m] = s/(lmd[i] - t)

#                    acf[ix + i] = qi # a.c.f. estimate

#                    err_lmd[ix + i] = d

#                    err_X[ix + i] = dX[ix + i]*qx/(1 - qx)

#                          abs(err_lmd[i]), abs(err_lmd[m + i]), \
#                          abs(err_X[i]), abs(err_X[m + i]), acf[i]))

#                        print(msg % (j, lmd[ix + i], err_X[ix + i]))

#                        print(msg % (j, lmd[k], err_X[k]))

#                self.errors_val = numpy.concatenate \
#                    ((self.errors_val, err_lmd[0, ix : ix + lcon]))
#                self.errors_vec = numpy.concatenate \
#                    ((self.errors_vec, err_X[0, ix : ix + lcon]))

#                self.errors_val = numpy.concatenate \
#                    ((self.errors_val, err_lmd[0, jx - rcon : jx]))
#                self.errors_vec = numpy.concatenate \
#                    ((self.errors_vec, err_X[0, jx - rcon : jx]))

#                    acf[m + i] = acf[m + i + shift_left]

#                    err_lmd[m + i] = err_lmd[m + i + shift_left]

#                    err_X[m + i] = err_X[m + i + shift_left]

#                    acf[m + i] = 1.0

#                    err_lmd[m + i] = -1.0

#                    err_X[m + i] = -1.0

#                    acf[m + i] = acf[m + i - shift_right]

#                    err_lmd[m + i] = err_lmd[m + i - shift_right]

#                    err_X[m + i] = err_X[m + i - shift_right]

#                    acf[m + i] = 1.0

#                    err_lmd[m + i] = -1.0

#                    err_X[m + i] = -1.0

        cnv = solver.convergence_data('c', i)
        if cnv:
            return cnv

        if what[0] == 'c':
            return self.cnv[which]

        if what[0] == 'r':
            return self.res[which]
        if what[0 : 2] == 'va':
            return self.err_lmd[:, which]
        if what[0 : 2] == 've':
            return self.err_X[:, which]

#        l = solver.convergence_data('next left')
#        r = solver.convergence_data('next right')
#        print('iterating vectors %d to %d out of %d' % (l, r, n))

#        elif what.find('next') > -1:
#            if what.find('left') > -1:
#                return self.ix
#            elif what.find('right') > -1:
#                return self.ix + self.nx - 1

opt.res_tol = 1e-10

##d = numpy.asarray([i/5 + 1 for i in range(m)])
##a = numpy.ones((m, n))
##s = numpy.linalg.norm(a[:, 0])
##a[:, 0] /= s
##for i in range(1, n):
##    a[:, i] = d*a[:, i - 1]
##    s = numpy.linalg.norm(a[:, i])
##    a[:, i] /= s

#            u = u.astype(self.type)

#        if type_x is not self.type:
#            mixed_types = True
#            u = u.astype(self.type)
#        else:
#            mixed_types = False
#        z = numpy.dot(u, self.a.T)
#        if mixed_types:
#            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
#        else:
#            y.data()[:,:] = numpy.dot(z, self.a)

#    s = numpy.random.rand(k).astype(dt)
#    s = numpy.sort(s)
#    t = numpy.ones(k)*s[0]
##    sigma = lambda t: 2**(-alpha*t).astype(dt)
#    s = sigma(k*(s - t))
##    s = 2**(-alpha*k*(s - t)).astype(dt)

#def random_svd(m, n, alpha):
#    k = min(m, n)
#    u = numpy.random.randn(m, k).astype(numpy.float32)
#    v = numpy.random.randn(n, k).astype(numpy.float32)
#    s = numpy.random.rand(k).astype(numpy.float32)
#    u, r = numpy.linalg.qr(u)
#    v, r = numpy.linalg.qr(v)
#    s = numpy.sort(s)
#    t = numpy.ones(k)*s[0]
#    s = 2**(-alpha*k*(s - t)).astype(numpy.float32)
#    a = numpy.dot(u*s, v.transpose())
#    return s, u, v, a

#        if type_x is not self.type:
#            mixed_types = True
#            u = u.astype(self.type)
#        else:
#            mixed_types = False
#        z = numpy.dot(u, self.a.T)
#        #z = numpy.dot(x.data(), self.a.T)
#        if mixed_types:
#            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
#        else:
#            y.data()[:,:] = numpy.dot(z, self.a)
