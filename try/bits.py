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
