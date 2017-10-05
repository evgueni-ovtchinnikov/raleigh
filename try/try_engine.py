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

opt = Options()
opt.block_size = 2
solver = Engine(opt, problem = 's')

n = 10
m = opt.block_size
k = solver.blocks()
w = numpy.ndarray((k,n,m))
w.fill(1.0)
mm = m + m
rr = numpy.ndarray((3, mm, mm))
rr.fill(-1)

while True:
    solver.move()
    rci = solver.rci()
    job = rci.job
    nx = rci.nx
    ny = rci.ny
    kx = rci.kx
    ky = rci.ky
    jx = rci.jx
    jy = rci.jy
    i = rci.i
    j = rci.j
    k = rci.k
    alpha = rci.alpha
    beta = rci.beta
    print('job %d' % job)
    if job < 0:
        break
    if job == APPLY_A:
        w[ky, :, jy : jy + nx] = opA(n, w[kx, :, jx : jx + nx])
        for j in range(jy, jy + nx):
            print(w[ky,:,j])
    if job == APPLY_B:
        w[ky, :, jy : jy + nx] = opB(n, w[kx, :, jx : jx + nx])
        for j in range(jy, jy + nx):
            print(w[ky,:,j])
    elif job == COMPUTE_XY:
        if nx < 1 or ny < 1:
            continue
        rr[k, j : j + ny, i : i + nx] = \
            alpha*numpy.dot( \
                w[ky, :, jy : jy + ny].transpose(), 
                w[kx, :, jx : jx + nx]) + \
            beta*rr[k, j : j + ny, i : i + nx]
        print(rr[k,:,:])
