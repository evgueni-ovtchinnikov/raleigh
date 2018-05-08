# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:09:33 2018

@author: wps46139
"""

import ctypes
import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla
import time

from sys import platform

print(platform)

if platform == 'win32':
    mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
else:
    mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)

print(mkl.mkl_get_max_threads())
#print(mkl.CblasColMajor)

CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113

numpy.random.seed(1) # make results reproducible

m = 6400
n = 40000
k = 32
l = 2500
dt = numpy.float32

def mydot(mkl, n, u, v):
    dot = mkl.cblas_sdot
    dot.restype = ctypes.c_float
    mkl_n = ctypes.c_int(n)
    ptr_u = ctypes.c_void_p(u.ctypes.data)
    ptr_v = ctypes.c_void_p(v.ctypes.data)
    mkl_incu = ctypes.c_int(1)
    mkl_incv = ctypes.c_int(1)
    r = dot(mkl_n, ptr_u, mkl_incu, ptr_v, mkl_incv)
    return r

def test1():

    u = numpy.random.randn(k, n).astype(dt)
    v = numpy.random.randn(k, n).astype(dt)
    w = numpy.random.randn(k, n).astype(dt)
    q = numpy.random.randn(k - 2, k - 2).astype(dt)
    s = numpy.zeros((k,))
    i = 1
    j = k - 1

    if dt == numpy.float32:
        gemm = mkl.cblas_sgemm
        axpy = mkl.cblas_saxpy
        copy = mkl.cblas_scopy
        scal = mkl.cblas_sscal
        norm = mkl.cblas_snrm2
        norm.restype = ctypes.c_float
        dot = mkl.cblas_sdot
        dot.restype = ctypes.c_float
        mkl_alpha = ctypes.c_float(1.0)
        mkl_beta = ctypes.c_float(0.0)
    else:
        gemm = mkl.cblas_dgemm
        axpy = mkl.cblas_daxpy
        copy = mkl.cblas_dcopy
        scal = mkl.cblas_dscal
        norm = mkl.cblas_dnrm2
        norm.restype = ctypes.c_double
        dot = mkl.cblas_ddot
        dot.restype = ctypes.c_double
        mkl_alpha = ctypes.c_double(1.0)
        mkl_beta = ctypes.c_double(0.0)

    mkl_n = ctypes.c_int(n)
    mkl_k = ctypes.c_int(k - 2)
#    mkl_nk = ctypes.c_int(n*(k - 2))
    mkl_incu = ctypes.c_int(1)
    mkl_incv = ctypes.c_int(1)

#    print(type(u.ctypes.data))
#    ptr_u1 = ctypes.c_void_p(u.ctypes.data + 4)
#    ptr_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#    print(u[0,0], ptr_u[0])
#    return

    uij = u[i:j,:]
    vij = u[i:j,:]
#    ptr_u = uij.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#    ptr_v = vij.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#    print(u.ctypes.data, uij.ctypes.data)
#    print(ptr_u)
#    print(u[i,0])
#    uij[0,0] = 999
#    print(u[i,0])
#    print(ptr_u[0])
    ptr_u = ctypes.c_void_p(u.ctypes.data + 4*n)
#    ptr_u = ctypes.c_void_p(uij.ctypes.data)
    ptr_v = ctypes.c_void_p(vij.ctypes.data)
    ptr_q = ctypes.c_void_p(q.ctypes.data)
    r = norm(mkl_n, ptr_v, mkl_incv)
    print(r)
    copy(mkl_n, ptr_v, mkl_incv, ptr_u, mkl_incu)
    r = norm(mkl_n, ptr_u, mkl_incu)
    print(r)
    r = dot(mkl_n, ptr_u, mkl_incu, ptr_v, mkl_incv)
    print(math.sqrt(r))

#    print('matrix dot...')
#    start = time.time()
#    for t in range(l):
#        numpy.dot(q.T, u[i:j,:], out = w[i:j,:])
#    stop = time.time()
#    time_dot = stop - start
#
    print('matrix axpy...')
    start = time.time()
    for t in range(l):
        numpy.dot(q.T, u[i:j,:], out = w[i:j,:])
#        w[:,:] = numpy.dot(q.T, u)
        w[i:j,:] += v[i:j,:]
#        w[:,:] = v + numpy.dot(q.T, u)
    stop = time.time()
    time_axpy = stop - start

    print('matrix axpy (mkl)...')
    start = time.time()
    for t in range(l):
        gemm(CblasColMajor, CblasNoTrans, CblasTrans, mkl_n, mkl_k, mkl_k, \
             mkl_alpha, ptr_u, mkl_n, ptr_q, mkl_k, mkl_beta, ptr_v, mkl_n)
    stop = time.time()
    time_axpy_mkl = stop - start
#    return

#    print('matrix copy...')
#    start = time.time()
#    for t in range(l):
#        u[i:j,:] = w[i:j,:]
#    stop = time.time()
#    time_copy = stop - start

# makes no difference from above
#    print('matrix copy...')
#    start = time.time()
#    for t in range(l):
##        numpy.copyto(u[i:j,:], w[i:j,:])
#        vu = numpy.reshape(u[i:j,:], ((j - i)*n,))
#        vw = numpy.reshape(w[i:j,:], ((j - i)*n,))
#        numpy.copyto(vu, vw)
##        vu[:] = vw
#    stop = time.time()
#    time_vcopy = stop - start

#    print('matrix copy (mkl)...')
#    start = time.time()
#    for t in range(l):
#        copy(mkl_nk, ptr_v, mkl_incv, ptr_u, mkl_incu)
#    stop = time.time()
#    time_copy_mkl = stop - start

# too slow
#    print('vector norms...')
#    start = time.time()
#    for t in range(l):
#        s = nla.norm(u, axis = 1)
#    stop = time.time()
#    time_norm = stop - start
##    print(s)

    print('vector dots...')
    start = time.time()
    for t in range(l):
        for i in range(k):
            s[i] = numpy.dot(u[i,:], u[i,:])
#            s[i] = numpy.dot(u[i,:], v[i,:])
    s = numpy.sqrt(abs(s))
    stop = time.time()
    time_dots = stop - start
#    print(s)

    print('vector dots (mkl)...')
    start = time.time()
    for t in range(l):
        pu = u.ctypes.data
#        pv = v.ctypes.data
        for i in range(k):
#            ui = u[i,:]
#            vi = v[i,:]
#            ptr_ui = ctypes.c_void_p(ui.ctypes.data)
#            ptr_vi = ctypes.c_void_p(vi.ctypes.data)
#            ptr_ui = ctypes.c_void_p(u.ctypes.data + 4*n*i)
#            ptr_vi = ctypes.c_void_p(v.ctypes.data + 4*n*i)
            ptr_ui = ctypes.c_void_p(pu)
#            ptr_vi = ctypes.c_void_p(pv)
            s[i] = dot(mkl_n, ptr_ui, mkl_incu, ptr_ui, mkl_incv)
#            s[i] = dot(mkl_n, ptr_ui, mkl_incu, ptr_vi, mkl_incv)
            pu += 4*n
#            pv += 4*n
    s = numpy.sqrt(abs(s))
    stop = time.time()
    time_dots_mkl = stop - start
#    print(s)
#    start = time.time()
##    ui = u[i,:]
##    vi = v[i,:]
##    ptr_u = ui.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
##    ptr_v = vi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#    for i in range(k):
#        ui = u[i,:]
#        vi = v[i,:]
#        ptr_ui = ctypes.c_void_p(ui.ctypes.data)
#        ptr_vi = ctypes.c_void_p(vi.ctypes.data)
##        ptr_ui = ui.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
##        ptr_vi = vi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#        for t in range(l):
#            s[i] = dot(mkl_n, ptr_ui, mkl_incu, ptr_vi, mkl_incv)
## checking if dot is really called each time and not once!
##            if t % 2:
##                s[i] = t + dot(mkl_n, ptr_ui, mkl_incu, ptr_vi, mkl_incv)
##            else:
##                s[i] = t + dot(mkl_n, ptr_u, mkl_incu, ptr_v, mkl_incv)
#    s = numpy.sqrt(abs(s))
#    stop = time.time()
#    time_dots_mkl = stop - start
##    print(s)

    print('vector axpys...')
    start = time.time()
    for t in range(l):
        for i in range(k):
            v[i,:] += s[i]*u[i,:]
    stop = time.time()
    time_axpys = stop - start

    print('vector axpys (mkl)...')
    start = time.time()
    for t in range(l):
        pu = u.ctypes.data
        pv = v.ctypes.data
        for i in range(k):
#            ui = u[i,:]
#            vi = v[i,:]
#            ptr_ui = ctypes.c_void_p(ui.ctypes.data)
#            ptr_vi = ctypes.c_void_p(vi.ctypes.data)
#            ptr_ui = ctypes.c_void_p(u.ctypes.data + 4*n*i)
#            ptr_vi = ctypes.c_void_p(v.ctypes.data + 4*n*i)
            ptr_ui = ctypes.c_void_p(pu)
            ptr_vi = ctypes.c_void_p(pv)
            if dt == numpy.float32:
                mkl_s = ctypes.c_float(s[i])
            else:
                mkl_s = ctypes.c_double(s[i])
            axpy(mkl_n, mkl_s, ptr_ui, mkl_incu, ptr_vi, mkl_incv)
            pu += 4*n
            pv += 4*n
    stop = time.time()
    time_axpys_mkl = stop - start

    print('vectors scaling...')
    start = time.time()
    for t in range(l):
        for i in range(k):
            if s[i] != 0.0:
                v[i,:] *= 1.0/s[i]
    stop = time.time()
    time_scal = stop - start

    print('vectors scaling (mkl)...')
    start = time.time()
    for t in range(l):
        pu = u.ctypes.data
        for i in range(k):
#            ui = u[i,:]
#            ptr_ui = ctypes.c_void_p(ui.ctypes.data)
#            ptr_ui = ctypes.c_void_p(u.ctypes.data + 4*n*i)
            ptr_ui = ctypes.c_void_p(pu)
            if s[i] != 0.0:
                if dt == numpy.float32:
                    mkl_s = ctypes.c_float(1.0/s[i])
                else:
                    mkl_s = ctypes.c_double(1.0/s[i])
                scal(mkl_n, mkl_s, ptr_ui, mkl_incu)
            pu += 4*n
    stop = time.time()
    time_scal_mkl = stop - start

# too slow
#    print('vector dots via numpy.sum...')
#    start = time.time()
#    for t in range(l):
#        s = numpy.sum(u*u, axis = 1)
#    s = numpy.sqrt(s)
#    stop = time.time()
#    time_sum = stop - start
##    print(s)

#    print('dot time: %.1e' % time_dot)
    print('axpy time: %.1e %.1e' % (time_axpy, time_axpy_mkl))
#    print('mkl copy time: %.1e' % time_copy_mkl)
#    print('copy time: %.1e %.1e' % (time_copy, time_copy_mkl))
#    print('norm time: %.1e' % time_norm)
    print('dots time: %.1e %.1e' % (time_dots, time_dots_mkl))
    print('axpys time: %.1e %.1e' % (time_axpys, time_axpys_mkl))
    print('scaling time: %.1e %.1e' % (time_scal, time_scal_mkl))
#    print('sum time: %.1e' % time_sum)

test1()

print('done')
