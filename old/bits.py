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
