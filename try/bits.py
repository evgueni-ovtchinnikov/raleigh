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
