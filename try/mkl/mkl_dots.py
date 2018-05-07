import ctypes
import numpy
import time
from sys import platform
if platform == 'win32':
    mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
else:
    mkl = ctypes.CDLL('mkl_rt.so', mode = ctypes.RTLD_GLOBAL)

numpy.random.seed(1) # make results reproducible

n = 40000
k = 32
l = 2500
dt = numpy.float32

u = numpy.random.randn(k, n).astype(dt)
v = numpy.random.randn(k, n).astype(dt)
s = numpy.zeros((k,))

dot = mkl.cblas_sdot
dot.restype = ctypes.c_float

mkl_n = ctypes.c_int(n)
inc = ctypes.c_int(1)

start = time.time()
for t in range(l):
    pu = u.ctypes.data
    pv = v.ctypes.data
    for i in range(k):
        ptr_ui = ctypes.c_void_p(pu)
        ptr_vi = ctypes.c_void_p(pv)
        s[i] = dot(mkl_n, ptr_ui, inc, ptr_vi, inc)
        pu += 4*n
        pv += 4*n
stop = time.time()
print('time: %.1e' % (stop - start))

start = time.time()
pu = u.ctypes.data
pv = v.ctypes.data
for i in range(k):
    ptr_ui = ctypes.c_void_p(pu)
    ptr_vi = ctypes.c_void_p(pv)
    for t in range(l):
        s[i] = dot(mkl_n, ptr_ui, inc, ptr_vi, inc)
    pu += 4*n
    pv += 4*n
stop = time.time()
print('time: %.1e' % (stop - start))
