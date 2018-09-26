# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:37:46 2017

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

#            if self.iteration == 102:
#                print(lmdx)
#                print(da)
#                print(db)

#s, u, v, a = random_matrix_for_svd(m, n, sigma, numpy.float32)
#numpy.save('rand10K4K.npy', a)
#a = numpy.load('rand10K4K.npy')

#            z = numpy.dot(u.astype(self.type), self.a.T)

#            z = numpy.dot(u, self.a.T)

            if self.a.flags['C_CONTIGUOUS']:
                print('a is C contiguous')
            b = conjugate(self.a)
            if b.flags['C_CONTIGUOUS']:
                print('a.T is C contiguous')

            if z.flags['C_CONTIGUOUS']:
                print('z is C contiguous')

            if y.data().flags['C_CONTIGUOUS']:
                print('y is C contiguous')

#        elif m < 2:
#            if verb > -1:
#                print('Block size 1 too small, will use 2 instead')
#            m = 2

#        l = int(round(r*m))
#        if l == 0 and r > 0.0:
#            l = 1
#        if l == m and r < 1.0:
#            l = m - 1

#                print(dlmd[ix : ix + nx, rec - 1])

#                elif res[k] < delta and acf[0, k] > acf[1, k]:

#            if lcon + rcon > 0:
#                if left < 0:
#                    shift_left_max = lcon
#                else:
#                    shift_left_max = max(0, left_total - self.lcon - leftX)
#                if right < 0:
#                    shift_right_max = rcon
#                else:
#                    shift_right_max = max(0, right_total - self.rcon - rightX)
#                if lcon + rcon <= ny:
#                    shift_left = lcon
#                    shift_right = rcon
#                else:
#                    shift_left = min(lcon, int(round(lr_ratio*ny)))
#                    shift_right = min(rcon, ny - shift_left)
#                shift_left = min(shift_left, shift_left_max)
#                shift_right = min(shift_right, shift_right_max)
#            else:
#                shift_left = 0
#                shift_right = 0

                #rightX_new = rightX + shift_right

                #left_block_size_new = ix + leftX + rightX - rightX_new

            #print('shifts:', shift_left, shift_right)

#    if m >= n:
#        sigma, u, vt = partial_svd(a, opt, vt, nsv)
#    else:
#        sigma, u, vt = partial_svd(a, opt, u, nsv)

def partial_svd0(a, opt, vtc = None, nsv = -1):
    m, n = a.shape
    if m >= n:
        vt = compute_right(a, opt, vtc, nsv)
        sigma, u, vt = compute_left(a, vt)
        return sigma, u, vt
    else:
        b = conjugate(a)
        if vtc is not None:
            vtc = conjugate(vtc)
        vt = compute_right(b, opt, vtc, nsv)
        sigma, u, vt = compute_left(b, vt)
        return sigma, conjugate(vt), conjugate(u)

#                        abs(res[k] - res_prev[k]) < 0.01*res[k]) and \

        #res_prev = -numpy.ones((m,), dtype = numpy.float32)

#        lmdB, Q = sla.eigh(XBX)
#        print(lmdB)
            
            #res_prev[ix : ix + nx] = res[ix : ix + nx]

                    #print('%d %e' % (ix + i, dX[ix + i]))

#            print('YY:')
#            print(Y.dots(Y))
            
#                print('Z:')
#                print(Z.dots(Z))
#                print(AZ.dots(AZ))
#                print(BZ.dots(BZ))

                # TODO: avoid division by zero
#                print('Num:')
#                print(Num)
#                print('Den:')
#                print(Den)
#                select = abs(Den) <= 1e-4*abs(Num)
#                Den[select] = 1e-4*Num[select]
#                Den[Num == 0.0] = 1.0

#                print('Den:')
#                print(Den)
#                print(Num[Den == 0.0])
#                Beta = Num/Den
#                print('Beta:')
#                print(Beta)

                    #B(Z, BZ)

#                    print('BY:')
#                    print(BY.dots(BY))
#                print('Y:')
#                print(Y.dots(Y))

#                print('YY:')
#                print(Y.dots(Y))
#                print('Y:')
#                print(Y.data())
#                print('BY:')
#                print(BY.data())
#                print('YBY:')
#                print(BY.dots(Y))
#                print('YBBY:')
#                print(BY.dots(BY))

#            lmdB, Q = sla.eigh(GB)
#            print('lmdB:')
#            print(lmdB)

#            GB = numpy.dot(U.T, U)
#            lmdB, Q = sla.eigh(GB)
#            print('lmdB:')
#            print(lmdB)

#            #B(Y, BY)
#            print('YBY:')
#            print(BY.dots(Y))
    
#            YBY = BY.dot(Y)
#            lmdy, Qy = sla.eigh(YAY, YBY)
#            print('lmdy:')
#            print(lmdy)
#
#            lmdxy, Q = sla.eigh(GA, GB, turbo=False)
#            print('lmdxy:')
#            print(lmdxy)

#            print('lmdy:')
#            print(lmdy)

#            print('lmdxy:')
#            print(lmdxy)

#            # estimate changes in eigenvalues and eigenvectors
#            lmdx = numpy.concatenate \
#                ((lmdxy[:leftX_new], lmdxy[nxy - rightX_new:]))
#            lmdy = lmdxy[leftX_new : nxy - rightX_new - 1]
#            QX = numpy.concatenate \
#                ((Q[:, :leftX_new], Q[:, nxy - rightX_new:]), axis = 1)
#            QYX = QX[nx:, :].copy()
#            lmdX = numpy.ndarray((1, nx_new))
#            lmdY = numpy.ndarray((ny, 1))
#            lmdX[0, :] = lmdx
#            lmdY[:, 0] = lmdy
#            Delta = (lmdY - lmdX)*QYX*QYX
#            dX[ix_new : ix_new + nx_new] = nla.norm(QYX, axis = 0)
#            if rec == RECORDS:
#                for i in range(rec - 1):
#                    dlmd[:, i] = dlmd[:, i + 1]
#            else:
#                rec += 1
#            dlmd[ix_new : ix_new + nx_new, rec - 1] = numpy.sum(Delta, axis = 0)

#            lft = max(leftX, leftX_new)
#            rgt = max(rightX, rightX_new)

#                    if dlmd[ix + i, rec - 1] == 0: # previous data not available
#                        continue

#def random_matrix_for_svd(m, n, u, v, sigma, dt):
#    k = min(m, n)
#    u = numpy.random.randn(m, k).astype(dt)
#    v = numpy.random.randn(n, k).astype(dt)
#    u, r = numpy.linalg.qr(u)
#    v, r = numpy.linalg.qr(v)
#    s = random_singular_values(k, sigma, dt)
#    x = numpy.arange(k)
#    plt.figure()
#    plt.plot(x, s)
#    plt.show()
#    a = numpy.dot(u*s, v.transpose())
#    return s, u, v, a

#from random_matrix_for_svd import random_singular_values, random_singular_vectors

LOAD = True
SAVE = False

#if LOAD:
#    u0 = numpy.load('C:/Users/wps46139/Documents/Data/PCA/u10K4K.npy')
#    v0 = numpy.load('C:/Users/wps46139/Documents/Data/PCA/v10K4K.npy')
#    m, n = u0.shape
#else:
#    # generate the matrix
#    m = 1000
#    n = 400
#    u0, v0 = random_singular_vectors(m, n, numpy.float32)
#    if SAVE:
#        numpy.save('u.npy', u0)
#        numpy.save('v.npy', v0)
#k = min(m, n)

#sigma0 = random_singular_values(k, f_sigma, numpy.float32)

#A = numpy.dot(u0*sigma0, v0.transpose())

            #print(delta_R)

#            H = abs(XAX - conjugate(XAX))
#            delta = numpy.amax(H)

#            if delta >= 0:
#                err_AX = delta
#                err_AX_rel = delta/numpy.amax(numpy.sqrt(AX.dots(AX)))
#            #err_AX = max(err_AX, delta) # too large
#            if verb > 1:
#                print('estimated error in AX (abs, rel): %e %e' \
#                % (err_AX, err_AX_rel))

#    #u = Vector(numpy.random.randn(n,k)) 
#    u = Vector(u)
#    v = Vector(numpy.zeros((m,k)))
#    
#    print(nla.norm(A.dot(u.data())))
#    arg = ((A, u, v))
##    mult(arg)
#    map(mult, arg) # v is 0! see below
##    mult((A, u, v)) # ok
#    print(v.norm())
#    #print(v.data())
#
#    arg = ()
#    for i in range(k):
#        ui = Vector(u.data()[:,i])
#        vi = Vector(v.data()[:,i])
#        arg += ((A, ui, vi),)
#    
##    [mult(x) for x in arg]
##    g = mymap(mult, arg)
##    for i in g:
##        pass
#
#    g = map(mult, arg)
#    print(g)
##    for i in g:
##        pass
#
##    pool = Pool(k)
##    g = pool.map(mult, arg)
##    pool.close()
##    pool.join()
#
#    print(v.norm())
#print(v.data())

##print('A:')
##print(A)
#print('u:')
#print(nla.norm(u))
#print('v:')
#print(nla.norm(v))
#print('w:')
#print(nla.norm(w))

#arg = ((A, u[:,0], v[:,0]), (A, u[:,1], v[:,1]))

#map(mult, arg)

#output = tuple(map(mult, arg))

#v[:,0] = output[0]
#v[:,1] = output[1]
#print(v)

#print(type(output))
#print(type(output[0]))
#print(output[0].shape)
#print(output[0])

#            if self.iteration > 50:
#                print('iteration: %d' % self.iteration)
#                print('old lmd:')
#                print(lmd[56])
##                print(lmd[ix : ix + nx])
#                print('new lmd:')
#                print(new_lmd[56 - ix])
##                print(new_lmd)
##                print('da:')
##                print(da)
##                print('da0:')
##                print(da0)

#            delta_R = numpy.amax(RX, axis = 0)

#            elif pro:
#                s = numpy.sqrt(BX.dots(BX))
#                delta_R /= s # numpy.amax(s)

#            n = X.dimension()

#            delta_R *= math.sqrt(n)

#                        eps = 1e-4*max(abs(new_lmd[i]), abs(lmd[ix + i]))
#                        if abs(delta) > eps:

#                        print(self.iteration, ix + i, lmd[ix + i], new_lmd[i])

#                    if err_X[0, ix + i] < 0:
#                        print(self.iteration)
#                        print(ix + i, dX[ix + i], qx)
#                        print(dlmd[ix + i - 2: ix + i + 2,:].T)

#            s = X.dimension()
#            s = math.sqrt(s)
#            eps = 1e-15
#            delta = err_AX*s
#            print(delta)

#                elif res[k] >= 0 and res[k] < max(delta_R[i], eps*norm_AX[k]) and \

#                elif res[k] >= 0 and res[k] < max(delta_R[nx - i - 1], eps*norm_AX[k]) and \

#                    W.select(Y.nvec())
#                    B(Y, W)

#            if self.iteration > 50:
#                print(self.iteration)
#                print(lmdx[leftX - 1], lmdx[leftX])
#                print(lmdy[0], lmdy[-1])

#            if self.iteration > 50:
#                print('iteration %d, next dlmd:' % self.iteration)
#                print(dlmd[ix : ix + nx, rec - 1])

#    if isinstance(a[0,0], complex):

#                       (U.transpose(), A[:k, k : n], lower = True)

#            A[i, i : n] -= np.dot(A[:i, i].transpose(), A[:i, i : n])

        #if i >= k and lmin < 0.5 and lmin/lmax <= 2*eps:

#    return sla.norm(np.dot(U.T, U), ord = 1)

    #lmax = sla.norm(A, ord = 1)

            if rv_err > 0.1*dlmd_av:
                print('restarting...')
                A(X, AX)
                if not std:
                    B(X, BX)
                if pro:
                    XAX = AX.dot(BX)
                else:
                    XAX = AX.dot(X)
                XBX = BX.dot(X)
                new_lmd, Q = sla.eigh(XAX, XBX, turbo=False)
                #print(new_lmd)
                W.select(nx)
                if not std:
                    BX.mult(Q, W)
                    W.copy(BX)
                X.mult(Q, W)
                W.copy(X)
                AX.mult(Q, W)
                W.copy(AX)
                nz = 0
                rec = 0
                last_restart = self.iteration

        last_restart = 0

            if not pro:
                A(X, Y)
                Y.add(X, -lmd[ix : ix + nx])
                t = Y.dots(Y)
                for i in range(nx):
                    ri = math.sqrt(r[i])
                    si = math.sqrt(s[i])
                    ti = math.sqrt(t[i])
                    if ti > si:
                        print(lmd[ix + i], ri, si, ti)

#            if (self.iteration + 1) % 10 == 0:
#                A(X, AX)
#                A(Z, AZ)
#            W.select(nx)
#            A(X, W)
#            W.add(AX, -1.0)
#            s = numpy.sqrt(W.dots(W))
#            print('AX error: %.1e' % numpy.amax(s))

#    p = u.dot(u)
#    q = p - numpy.eye(p.shape[0])
#    print(numpy.linalg.norm(q))
#    q = v.dot(v)
#    q = q - numpy.eye(q.shape[0])
#    print(numpy.linalg.norm(q))
#    
#    z = u.new_vectors(rcon)

#    A(u, w)
#    q = w.dot(u)
#    lmd, Q = sla.eigh(q, p)
#    u.mult(Q, z)
#    z.copy(u)
#    w.mult(Q, z)
#    z.copy(w)
#
#    q = w.dot(u)
#    p = q - numpy.diag(lmdu[nconu - rcon :])
#    print(numpy.linalg.norm(p))
#    u.mult(q, z)
#    z.add(w, -1.0)
#    t = z.dots(z)
#    t = numpy.sqrt(t)
#    w.add(u, -lmdu[nconu - rcon :])
#    s = w.dots(w)
#    s = numpy.sqrt(s)

#    print('     first pass             second pass')

#        print('  %.1e / %.1e      %.1e / %.1e         %.1e' % \
#        (abs(err_ui[0]), abs(err_ui[1]), abs(err_vi[0]), abs(err_vi[1]), sl[i]))

#        res_ui = res_u[nconu - rcon + i]
#        lmdv_i = lmdv[nconv - rcon + i]

#        res_vi = res_v[nconv - rcon + i]

#        print('%e %.1e  %.1e / %.1e      %e %.1e  %.1e / %.1e     %.1e' % \
#        (lmdu_i, res_ui, err_uik, err_uir, lmdv_i, res_vi, err_vik, err_vir, sr[i]))

#    q = v.dot(u)
#    u.mult(q, w)
#    w.add(v, -1.0)
#    t = w.dots(w)
#    t = numpy.sqrt(t)

                        #abs(dlmd[k, rec - 1]) > abs(dlmd[k, rec - 2]) and \
                        #abs(dlmd[k, rec - 1]) < 10*abs(dlmd[k, rec - 2]):

                    first_rec = max(0, rec - self.iteration//3 - 2)
#                    for r in range(rec - 1, first_rec, -1):

#        ptr_ui = ctypes.c_void_p(pu)
#        ptr_vi = ctypes.c_void_p(pv)

#        s[i] = dot(mkl_n, ptr_ui, inc, ptr_vi, inc)

            delta_R = numpy.ndarray((nx,))
            delta_R[0] = RX[1,0]
            delta_R[1:] = numpy.diag(RX, 1)
#            delta_R = numpy.amax(RX, axis = 0)

#            for i in range(nx):
#                for j in range(nx):
#                    if XAX[i,i] > 4*XAX[j,j] or XAX[i,i] < 0.25*XAX[j,j]:
#                        RX[i,j] = 0

#                    BXc.mult(Q, Y)
#                    Xc.mult(Q, Y)
#                W.add(Y, -1.0)

#            print('max(XBY): %.1e' % numpy.amax(Y.dot(X)))

#                print('max(ZBY): %.1e' % numpy.amax(Y.dot(Z)))

#                Z.mult(Beta, AZ)
#                Y.add(AZ, -1.0)

#                    BZ.mult(Beta, AZ)
#                    W.add(AZ, -1.0)

#            print('max(XBY): %.1e' % numpy.amax(Y.dot(X)))

#            X.mult(Q, W)
#            Y.add(W, -1.0)

#                BX.mult(Q, W)
#                BY.add(W, -1.0)

#                print('max(XcBX): %.1e' % numpy.amax(X.dot(Xc)))

#                Xc.mult(Q, W)
#                Y.add(W, -1.0)

#                    BXc.mult(Q, W)
#                    BY.add(W, -1.0)

#            print('max(XBY): %.1e' % numpy.amax(XBY))

#                AY.mult(QYX, Z)
#                W.add(Z, 1.0) # W = AX*QXX + AY*QYX

#                    AX.mult(QXZ, AZ)
#                else:
#                    AZ.zero()

#            if nz > 0:
#                AZ.add(Z, 1.0) # AZ = AX*QXZ + AY*QYZ

#                    BY.mult(QYX, Z)
#                    W.add(Z, 1.0)

#                        BX.mult(QXZ, BZ)
#                    else:
#                        BZ.zero()

#                if nz > 0:
#                    BZ.add(Z, 1.0)

#                Y.mult(QYX, Z)
#                W.add(Z, 1.0) # W = X*QXX + Y*QYX

#                    X.mult(QXZ, Z)
#                else:
#                    Z.zero()
#            if nz > 0:
#                Z.add(Y, 1.0, QYZ)

#                W.select(nz)
#                Y.mult(QYZ, W)
#                Z.add(W, 1.0) # Z = X*QXZ + Y*QYZ

                #AZ = AY

#            Xc.multiply(Q, W)
#            X.add(W, -1.0)

#            if not std:
#                Q = Y.dot(BX)
#            else:
#                Q = Y.dot(X)

#                if not std:
#                    Q = numpy.dot(Gci, Y.dot(BXc))
#                else:
#                    Q = numpy.dot(Gci, Y.dot(Xc))

                    #BZ = BY

#        XAX = 0.5*(XAX + conjugate(XAX))
#        print(XAX)

#        print(lmdx)

#        AZ = None
#        BZ = None

#            print(delta_R)
#            print(numpy.sqrt(AX.dots(AX)))

#            print('Y.nvec: %d' % Y.nvec())
#            print('W.nvec: %d' % W.nvec())

#                    print(ind)
#                    print(math.sqrt(last_piv))

##        data = numpy.ones((nv, n), dtype = self.__data.dtype)
##        return Vectors(data)

        #data = numpy.zeros((nv, n), dtype = self.__data.dtype)

        #return Vectors(data)

        #a = numpy.zeros((m, n), dtype = self.__data.dtype)

        #return Vectors(a)

#                print(type(self.__data.ctypes.data))
#                print(type(ind[0]))
#                print(type(vsize))

#                    print(type(data_u))

#            q = numpy.ndarray((m, k), dtype = self.__data.dtype)

#            self.__gemm(CblasColMajor, Trans, CblasNoTrans, \
#                mkl_k, mkl_m, mkl_n, \
#                self.__mkl_one, ptr_u, mkl_n, ptr_v, mkl_n, \
#                self.__mkl_zero, ptr_q, mkl_k)
##            print(nla.norm(t - q.T))
#            qt[:,:] = q.T
#            return qt
#            return q.T

#        print(q.flags)

#            print(n, m, self.nvec(), f, fs)

#            pf_q = ctypes.cast(ptr_q, ctypes.POINTER(ctypes.c_float))
#            print(pf_q[0])
#            print(pf_q[1])
#            print(pf_q[2])

#            print('---')
#            print(nla.norm(output.__data[f : f + m, :]))
#            w = output.__data[f : f + m, :].copy()
#            print(nla.norm(w[0,:]))
#            numpy.dot(q.T, self.data(), out = output.__data[f : f + m, :])
#            print(nla.norm(w - output.__data[f : f + m, :]))

#import numpy.linalg as nla

#from sys import platform

#print(platform)

#    if platform == 'win32':
#        mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
#    else:
#        mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)

#    CblasColMajor = 102
#    CblasNoTrans = 111
#    CblasTrans = 112
#    CblasConjTrans = 113
#    print('Using %d MKL threads' % mkl.mkl_get_max_threads())

#            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)

#            y.data()[:,:] = numpy.dot(z, self.a)

#            if dt == numpy.float32:
#                self.__dsize = 4
#                self.__gemm = mkl.cblas_sgemm
#                self.__axpy = mkl.cblas_saxpy
#                self.__copy = mkl.cblas_scopy
#                self.__scal = mkl.cblas_sscal
#                self.__norm = mkl.cblas_snrm2
#                self.__norm.restype = ctypes.c_float
#                self.__inner = mkl.cblas_sdot
#                self.__inner.restype = ctypes.c_float
#                self.__mkl_one = ctypes.c_float(1.0)
#                self.__mkl_zero = ctypes.c_float(0.0)
#            elif dt == numpy.float64:
#                self.__dsize = 8
#                self.__gemm = mkl.cblas_dgemm
#                self.__axpy = mkl.cblas_daxpy
#                self.__copy = mkl.cblas_dcopy
#                self.__scal = mkl.cblas_dscal
#                self.__norm = mkl.cblas_dnrm2
#                self.__norm.restype = ctypes.c_double
#                self.__inner = mkl.cblas_ddot
#                self.__inner.restype = ctypes.c_double
#                self.__mkl_one = ctypes.c_double(1.0)
#                self.__mkl_zero = ctypes.c_double(0.0)
#            elif dt == numpy.complex64:
#                self.__dsize = 8
#                self.__gemm = mkl.cblas_cgemm
#                self.__axpy = mkl.cblas_caxpy
#                self.__copy = mkl.cblas_ccopy
#                self.__scal = mkl.cblas_cscal
#                self.__norm = mkl.cblas_scnrm2
#                self.__norm.restype = ctypes.c_float
#                self.__inner = mkl.cblas_cdotc_sub
#                self.__cmplx_val = numpy.zeros((2,), dtype = numpy.float32)
#                self.__cmplx_one = numpy.zeros((2,), dtype = numpy.float32)
#                self.__cmplx_one[0] = 1.0
#                self.__cmplx_zero = numpy.zeros((2,), dtype = numpy.float32)
#                self.__mkl_one = ctypes.c_void_p(self.__cmplx_one.ctypes.data)
#                self.__mkl_zero = ctypes.c_void_p(self.__cmplx_zero.ctypes.data)
#            elif dt == numpy.complex128:
#                self.__dsize = 16
#                self.__gemm = mkl.cblas_zgemm
#                self.__axpy = mkl.cblas_zaxpy
#                self.__copy = mkl.cblas_zcopy
#                self.__scal = mkl.cblas_zscal
#                self.__norm = mkl.cblas_dznrm2
#                self.__norm.restype = ctypes.c_double
#                self.__inner = mkl.cblas_zdotc_sub
#                self.__cmplx_val = numpy.zeros((2,), dtype = numpy.float64)
#                self.__cmplx_one = numpy.zeros((2,), dtype = numpy.float64)
#                self.__cmplx_one[0] = 1.0
#                self.__cmplx_zero = numpy.zeros((2,), dtype = numpy.float64)
#                self.__mkl_one = ctypes.c_void_p(self.__cmplx_one.ctypes.data)
#                self.__mkl_zero = ctypes.c_void_p(self.__cmplx_zero.ctypes.data)
#            else:
#                raise ValueError('data type %s not supported' % repr(dt))

#                    self.__copy(mkl_n, ptr_u, mkl_inc, ptr_v, mkl_inc)

#                    self.__scal(mkl_n, mkl_s, ptr_u, mkl_inc)

#            vsize = self.__dsize * n
#            data_u = other.__data.ctypes.data + other.__selected[0] * vsize
#            data_v = self.__data.ctypes.data + self.__selected[0] * vsize

#            fs = self.__selected[0]

#            vsize = self.__cblas.dsize * n
#            data_u = output.__data.ctypes.data + f * vsize
#            data_v = self.__data.ctypes.data + fs * vsize

#            fu, m = other.__selected

#            data_u = other.__data.ctypes.data + fu * vsize
#            data_v = self.__data.ctypes.data + f * vsize

#                    data_u = other.__data.ctypes.data + (fu + i) * vsize
#                    data_v = self.__data.ctypes.data + (f + i) * vsize
#                    data_u = other.data().ctypes.data + i*vsize
#                    data_v = self.data().ctypes.data + i*vsize

#from raleigh.ndarray.vectors import Vectors
#from raleigh.ndarray.numpy_vectors import Vectors

#        self.type = type(a[0,0])

#    def apply(self, x, y, transp = False):
#        u = x.data()
#        type_x = type(u[0,0])
#        mixed_types = type_x is not self.type
#        if mixed_types:
#            u = u.astype(self.type)
#        if transp:
#            v = numpy.dot(u, self.a)
#        else:
#            v = numpy.dot(u, self.a.T)
#        if mixed_types:
#            v = v.astype(type_x)
#        y.data()[:,:] = v

#    def apply(self, x, y, transp = False):
#        u = x.data()
#        type_x = type(u[0,0])
#        mixed_types = type_x is not self.type
#        if mixed_types:
#            u = u.astype(self.type)
#        if transp:
#            w = numpy.dot(u, self.a)
#            v = numpy.dot(w, self.a.T)
#        else:
#            w = numpy.dot(u, self.a.T)
#            v = numpy.dot(w, self.a)
#        if mixed_types:
#            v = v.astype(type_x)
#        y.data()[:,:] = v
##        if mixed_types:
##            z = numpy.dot(u.astype(self.type), conjugate(self.a))
##            y.data()[:,:] = numpy.dot(z, self.a).astype(type_x)
##        else:
##            z = numpy.dot(u, conjugate(self.a))
##            y.data()[:,:] = numpy.dot(z, self.a)

def compute_right(a, transp, opt, nsv, vtc = None):
    if transp:
        n, m = a.shape
    else:
        m, n = a.shape
    if vtc is None:
        v = Vectors(n)
    else:
        v = Vectors(vtc, with_mkl = False)
    operator = OperatorSVD(a)
    problem = Problem(v, lambda x, y: operator.apply(x, y, transp))
    solver = Solver(problem)
    solver.solve(v, opt, which = (0, nsv))
    vt = v.data()
    return vt

def compute_left(a, vt):
    v = conjugate(vt)
    u = numpy.dot(a, v)
    vv = numpy.dot(conjugate(v), v)
    uu = -numpy.dot(conjugate(u), u)
    lmd, x = scipy.linalg.eigh(uu, vv, turbo = False)
    u = numpy.dot(u, x)
    v = numpy.dot(v, x)
    sigma = numpy.linalg.norm(u, axis = 0)
    u /= sigma
    return sigma, u, conjugate(v)

def partial_svd_old(a, opt, nsv = -1, uc = None, vtc = None):
    m, n = a.shape
    if m >= n:
        vt = compute_right(a, False, opt, nsv, vtc)
        sigma, u, vt = compute_left(a, vt)
        return sigma, u, vt
    else:
        print('transposing...')
        b = conjugate(a)
        if uc is not None:
            vtc = conjugate(uc)
        else:
            vtc = None
#        u = compute_right(b, opt, nsv, vtc)
        u = compute_right(a, True, opt, nsv, vtc)
        sigma, vt, u = compute_left(b, u)
        return sigma, conjugate(u), conjugate(vt)

try:
    from raleigh.ndarray.mkl import mkl, Cblas
    HAVE_MKL = True
except:
    HAVE_MKL = False

#            print(n, self.nvec(), m, mkl_k, mkl_m)

#            self.__cblas.mkl_one, ptr_v, mkl_n, ptr_q, mkl_m, \

#cudaMemcpyHostToHost          =   c_int(0)
#cudaMemcpyHostToDevice        =   c_int(1)
#cudaMemcpyDeviceToHost        =   c_int(2)
#cudaMemcpyDeviceToDevice      =   c_int(3)

#cuda = CDLL(cuda_path + '/cudart64_70.dll', mode = RTLD_GLOBAL)
#cublas = CDLL('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0\\bin\\cublas64_70.dll', mode = RTLD_GLOBAL)

#cuda_malloc = cuda.cudaMalloc
#cuda_malloc.argtypes = [POINTER(POINTER(c_ubyte)), c_int]
#cuda_malloc.restype = c_int
#
#cuda_free = cuda.cudaFree
#cuda_free.restype = c_int
#
#cuda_memcpy = cuda.cudaMemcpy
#cuda_memcpy.restype = c_int

#print(cuda_malloc(byref(dev_v), size))

#print(cuda_memcpy(dev_v, c_void_p(v.ctypes.data), size, cudaMemcpyHostToDevice))

#cuda_free(dev_v)

##cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/bin'
#cublas = CDLL(cuda.cuda_path + '/cublas64_70.dll', mode = RTLD_GLOBAL)
#
#cublas_create = cublas.cublasCreate_v2
#cublas_create.argtypes = [POINTER(POINTER(c_ubyte))]
#cublas_create.restype = c_int
#
#cublas_destroy = cublas.cublasDestroy_v2
##cublas_destroy.argtypes = [POINTER(c_ubyte)]
#cublas_destroy.restype = c_int

#handle = POINTER(ctypes.c_ubyte)()
#print(cublas.create(ctypes.byref(handle)))

#norm = cublas.cublas.cublasSnrm2_v2
#norm.restype = ctypes.c_int

#        a = self.__data[iv : iv + nv, :]
#        a[0,0] = 1.0
#        i = 1
#        while 2*i < m:
#            a[i : 2*i, :i] = a[:i, :i]
#            a[:i, i : 2*i] = a[:i, :i]
#            a[i : 2*i, i : 2*i] = -a[:i, :i]
#            i *= 2
#        k = i
#        j = 2*i
#        if j > n:
#            for i in range(k, m):
#                a[i, i] = 1.0
#            return
#        while j <= n:
#            a[:k, i : j] = a[:k, :i]
#            i, j = j, 2*j
#        j = i//2
#        a[k : m,   : j] = a[:(m - k), : j]
#        a[k : m, j : i] = -a[:(m - k), j : i]

#    def fill_orthogonal(self, m):

#        if n < m:
#            print('Warning: number of vectors too large, reducing')
#            m = n

#    if isinstance(a[0,0], complex):
#    if numpy.iscomplex(a).any():

#    def __to_mkl_float(self, v):
#        dt = self.data_type()
#        if dt == numpy.float32:
#            return ctypes.c_float(v)
#        elif dt == numpy.float64:
#            return ctypes.c_double(v)
#        elif dt == numpy.complex64 or dt == numpy.complex128:
#            self.__cblas.cmplx_val[0] = v.real
#            self.__cblas.cmplx_val[1] = v.imag
#            return ctypes.c_void_p(self.__cblas.cmplx_val.ctypes.data)
#        else:
#            raise ValueError('data type %s not supported' % repr(dt))

#            data_u = self.all_data().ctypes.data + i*vsize
#            data_v = other.all_data().ctypes.data + j*vsize
#            ptr_u = ctypes.c_void_p(data_u)
#            ptr_v = ctypes.c_void_p(data_v)

#                data_u = self.all_data().ctypes.data + int(ind[k])*vsize
#                data_v = other.all_data().ctypes.data + (j + k)*vsize
#                ptr_u = ctypes.c_void_p(data_u)
#                ptr_v = ctypes.c_void_p(data_v)

#                data_u = self.all_data().ctypes.data + (f + i)*vsize
#                ptr_u = ctypes.c_void_p(data_u)

#                mkl_s = self.__to_mkl_float(1.0/s[i])

#        if self.is_complex():
#            ptr_r = ctypes.c_void_p(self.__cblas.cmplx_val.ctypes.data)

#            data_u = self.all_data().ctypes.data + (iu + i)*vsize
#            data_v = other.all_data().ctypes.data + (iv + i)*vsize
#            ptr_u = ctypes.c_void_p(data_u)
#            ptr_v = ctypes.c_void_p(data_v)

#                self.__cblas.inner \
#                    (mkl_n, ptr_v, mkl_inc, ptr_u, mkl_inc, ptr_r)
#                res = self.__cblas.cmplx_val
#                v[i] = res[0] + 1j * res[1]

#        data_u = other.data().ctypes.data
#        data_v = self.data().ctypes.data
#        ptr_u = ctypes.c_void_p(data_u)
#        ptr_v = ctypes.c_void_p(data_v)

#        data_u = output.data().ctypes.data
#        data_v = self.data().ctypes.data
#        ptr_u = ctypes.c_void_p(data_u)
#        ptr_v = ctypes.c_void_p(data_v)

#            is_complex = isinstance(a[0,0], complex)
#            is_complex = numpy.iscomplex(a).any()

#        print(m, n, k, q.flags['C_CONTIGUOUS'])

#        data_u = output.data().ctypes.data
#        data_v = self.data().ctypes.data
#        ptr_u = ctypes.c_void_p(data_u)
#        ptr_v = ctypes.c_void_p(data_v)

#        if output.data().flags['C_CONTIGUOUS']:
#            #print('using optimized dot')
#            numpy.dot(self.data(), q.T, out = output.data())
#        else:
#            #print('using non-optimized dot')
#            output.data()[:,:] = numpy.dot(self.data(), q.T)

#        data_u = other.data().ctypes.data
#        data_v = self.data().ctypes.data
#        ptr_u = ctypes.c_void_p(data_u)
#        ptr_v = ctypes.c_void_p(data_v)

#            mkl_s = self.__to_mkl_float(s)

#                ptr_u = ctypes.c_void_p(data_u)
#                ptr_v = ctypes.c_void_p(data_v)

#                mkl_s = self.__to_mkl_float(s[i])

#                data_u += vsize
#                data_v += vsize

#    x = np.ones((n,), dtype = np.dtype(U[0,0]))

#def is_complex(v):
#    return type(v) is numpy.complex64 or type(v) is numpy.complex128

#    if numpy.iscomplex(a[0,0]): # lying!!!
#    if is_complex(a[0,0]):
#    if numpy.iscomplex(a).any():

#    if is_complex(a[0]):

#                s = self.__floats()
#                self.__cublas.dot \
#                    (self.__cublas.handle, n, ptr_v, inc, ptr_u, inc, s)

#                s = self.__float()
#                self.__cublas.dot \
#                    (self.__cublas.handle, n, ptr_v, inc, ptr_u, inc, ctypes.byref(s))
#                v[i] = s.value

#        if transp:
#            x.apply(self.a.T, y)
#        else:
#            x.apply(self.a, y)

#    w_cublas = cublasVectors(n, data_type = dt)
##    print(u_cublas.data())
##    print(w_cublas.data())

#    print('----\n testing cublasVectors.append...')
#    w_numpy.append(u_numpy)
#    start = time.time()
#    w_cublas.append(u_cublas)
#    cuda.synchronize()
#    stop = time.time()
#    elapsed = stop - start
#    t = nla.norm(w_cublas.data() - w_numpy.data())
#    print('error: %e' % t)
#    print('time: %.2e' % elapsed)
#    v_numpy.select(10, 5)
#    v_cublas.select(10, 5)
#    w_numpy.append(v_numpy)
#    w_cublas.append(v_cublas)
#    t = nla.norm(w_cublas.data() - w_numpy.data())
#    print('error: %e' % t)
#    v_numpy.select_all()
#    v_cublas.select_all()

        if m < 1:
            self.__mvec = MIN_INC
        else:
            self.__mvec = MIN_INC*((m - 1)//MIN_INC + 1)

#from raleigh.ndarray.cblas_algebra import Vectors, Matrix
#print('using mkl cblas...')

#from raleigh.cuda.cublas_algebra import Vectors
#from raleigh.cuda.cublas_algebra import Matrix

try:
    from raleigh.cuda.cublas_algebra import Vectors, Matrix0
    print('using cublas...')
except:

#from raleigh.ndarray.numpy_algebra import Vectors, Matrix
#from raleigh.algebra import Vectors, Matrix
#from raleigh.cuda.cublas_algebra import Vectors, Matrix

#    class Operator:
#        def __init__(self, array):
#            self.matrix = Matrix(array)
#        def apply(self, x, y, transp = False):
#            self.matrix.apply(x, y, transp)    

#        def __init__(self, array):
#            self.matrix = Matrix(array)

#            m, n = self.matrix.shape()

#                self.matrix.apply(x, z, transp = True)
#                self.matrix.apply(z, y)

#                self.matrix.apply(x, z)
#                self.matrix.apply(z, y, transp = True)

#    opSVD = OperatorSVD(a)

#class Operator:
#    def __init__(self, array):
#        self.matrix = Matrix(array)
#    def apply(self, x, y, transp = False):
#        self.matrix.apply(x, y, transp)
#
#class OperatorSVD:
#    def __init__(self, array):
#        self.matrix = Matrix(array)
#    def apply(self, x, y, transp = False):
#        m, n = self.matrix.shape()
#        k = x.nvec()
#        if transp:
#            z = Vectors(n, k, x.data_type())
#            self.matrix.apply(x, z, transp = True)
#            self.matrix.apply(z, y)
#        else:
#            z = Vectors(m, k, x.data_type())
#            self.matrix.apply(x, z)
#            self.matrix.apply(z, y, transp = True)

#data_root = 'C:/Users/wps46139/Documents/Data/PCA/'
#image_dir = data_root + 'lfw2/'
#image_dir = data_root + 'lfw_funneled/'

#diff = numpy.dot(v.T*sigma, u) - images
#err = nla.norm(diff, axis = 1)/nla.norm(images, axis = 1)
#print(nla.norm(err - err1)/nla.norm(err))

#            A[i, :], A[j, :] = A[j, :], A[i, :].copy()
#            A[:, i], A[:, j] = A[:, j], A[:, i].copy()

        #print('pivot: %e' % last_piv)
#        if A[i, i] <= eps:

#            return ind, n - i, last_piv
#        A[i, i] = math.sqrt(abs(A[i, i]))

#            return ind, n - i, last_piv

#    return ind, 0, A[n - 1, n - 1]**2

#        std_dev = math.sqrt(self.err.var())
#        print('std dev: %e' % std_dev)
#        k = (len(self.err[self.err > err_av + 3*std_dev]))
#        print('above average + 3*std_dev: %d' % k)
#        k = (len(self.err[self.err > 0.2]))
#        print('above 0.2: %d' % k)

  -e <err> , --svderr=<err>  acceptable svd approximation error [default: 0.1]

#        self.err_tol = 0.0

#        self.err_calc = PSVDErrorCalculator(a)
#        self.norms = self.err_calc.norms
#        self.err = self.norms
#        print('max data norm: %e' % numpy.amax(self.err))
#    def set_error_tolerance(self, err_tol):
#        self.err_tol = err_tol

#        self.err = self.err_calc.update_errors()
#        errs = (numpy.amax(self.err), numpy.amax(self.err/self.norms))
#        print('max err: abs %e, rel %e' % errs)
#        done = errs[1] <= self.err_tol or self.m > 0 and solver.rcon >= self.m

#    opt.stopping_criteria.set_how_many(block_size)
#opt.stopping_criteria.set_error_tolerance(err_tol)

