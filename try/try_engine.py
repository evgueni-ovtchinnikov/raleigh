import sys
sys.path.append('..')

from raleigh.engine import *

def opA(n, x):
    y = x.copy()
    for i in range(n):
        y[i,:] = (i + 1)*x[i,:]
    return y

def opB(n, x):
    return 2*x

def init_guess(n,m):
    w = numpy.ndarray((n,m))
    step = 1
    for j in range(m):
        for i in range(n):
            k = i//step + 1
            if k%2 == 1:
                w[i,j] = 1
            else:
                w[i,j] = -1
        step *= 2
    return w

opt = Options()
opt.block_size = 2
solver = Engine(opt, problem = 'g')

n = 8
m = opt.block_size
k = solver.blocks()
w = numpy.ndarray((k,n,m))
w[0,:,:] = init_guess(n,m)
mm = m + m
#rr = numpy.ndarray((3, mm, mm))
#rr.fill(-1)

while True:
    solver.move()
    rci = solver.rci()
    job = rci.job
    print(job)
    nx = rci.nx
    ny = rci.ny
    kx = rci.kx
    ky = rci.ky
    jx = rci.jx
    jy = rci.jy
    alpha = rci.alpha
    beta = rci.beta
    print('job %d' % job)
    if job < 0:
        break
    elif job == APPLY_A:
#        print('applying A:', kx, '->', ky)
        w[ky, :, jy : jy + nx] = opA(n, w[kx, :, jx : jx + nx])
        for j in range(jy, jy + nx):
            print(w[ky,:,j])
    elif job == APPLY_B:
        w[ky, :, jy : jy + nx] = opB(n, w[kx, :, jx : jx + nx])
        for j in range(jy, jy + nx):
            print(w[ky,:,j])
    elif job == COPY_VECTORS:
        if rci.i == 0:
            if kx != ky or jx > jy:
                w[ky, :, jy : jy + nx] = w[kx, :, jx : jx + nx]
            elif jx < jy:
                for i in range(nx):
                    j = nx - 1 - i
                    w[ky, :, jy + j] = w[kx, :, jx + j]
        else:
            for i in range(n):
                v = w[kx, i, solver.ind[:nx]]
                w[kx, i, :nx] = v
                if kx != ky:
                    v = w[ky, i, solver.ind[:nx]]
                    w[ky, i, :nx] = v
    elif job == COMPUTE_XY:
        if nx < 1 or ny < 1:
            continue
#        print('X:')
#        for j in range(jx, jx + nx):
#            print(w[kx,:,j])
#        print('Y:')
#        for j in range(jy, jy + ny):
#            print(w[ky,:,j])
        i = rci.i
        j = rci.j
        k = rci.k
        solver.rr[k, i : i + nx, j : j + ny] = \
            alpha*numpy.dot( \
                w[kx, :, jx : jx + nx].transpose(), 
                w[ky, :, jy : jy + ny]) + \
            beta*solver.rr[k, i : i + nx, j : j + ny]
#        solver.rr[k, j : j + ny, i : i + nx] = \
#            alpha*numpy.dot( \
#                w[ky, :, jy : jy + ny].transpose(), 
#                w[kx, :, jx : jx + nx]) + \
#            beta*solver.rr[k, j : j + ny, i : i + nx]
        print(solver.rr[k,:,:])
    elif job == COMPUTE_XQ or job == TRANSFORM:
#        print(nx, kx, jx)
#        print(ny, ky, jy)
        if ny < 1:
            continue
        if nx < 1:
            if job == TRANSFORM or beta == 1.0:
                continue
            w[ky, :, jy : jy + ny] *= beta
            continue
        i = rci.i
        j = rci.j
        k = rci.k
 #       print(i, j, k)
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
        if job == TRANSFORM:
            w[ky, :, jy : jy + ny] = \
                numpy.dot(w[kx, :, jx : jx + nx], \
                    solver.rr[k, i : i + nx, j : j + ny])
            w[kx, :, jx : jx + ny] = w[ky, :, jy : jy + ny]
        else:
            w[ky, :, jy : jy + ny] = \
                alpha*numpy.dot(w[kx, :, jx : jx + nx], \
                    solver.rr[k, i : i + nx, j : j + ny]) + \
                beta*w[ky, :, jy : jy + ny]

print(solver.lmd)
print(solver.rr[1,:m,:m])
print(w[0,:,:])
print(w[4,:,:])
print(w[3,:,:])
#print(w[6,:,:])
print(solver.rr[0,:m,:m])
print(solver.rr[1,:m,:m])
