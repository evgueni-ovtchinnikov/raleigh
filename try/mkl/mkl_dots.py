import ctypes
import numpy
import time
from sys import platform
if platform == 'win32':
    mkl = ctypes.CDLL('mkl_rt.dll', mode = ctypes.RTLD_GLOBAL)
else:
    mkl = ctypes.CDLL('libmkl_rt.so', mode = ctypes.RTLD_GLOBAL)
len = 128
version = ctypes.create_string_buffer(len)
mkl.mkl_get_version_string(version, ctypes.c_int(len))
print(version.value.decode('ascii'))
nthreads = mkl.mkl_get_max_threads
nthreads.restype = ctypes.c_int
nt = nthreads()
print('max mkl threads: %d' % nt)

numpy.random.seed(1) # make results reproducible

n = 100000
k = 32
l = 1000
dtype = numpy.float32
dsize = 4

u = numpy.random.randn(k, n).astype(dtype)
v = numpy.random.randn(k, n).astype(dtype)
s = numpy.zeros((k,), dtype = dtype)

dot = mkl.cblas_sdot
dot.restype = ctypes.c_float

mkl_n = ctypes.c_int(n)
inc = ctypes.c_int(1)

start = time.time()
for t in range(l):
    data_u = u.ctypes.data
    data_v = v.ctypes.data
    for i in range(k):
        ptr_ui = ctypes.c_void_p(data_u)
        ptr_vi = ctypes.c_void_p(data_v)
        s[i] = dot(mkl_n, ptr_ui, inc, ptr_vi, inc)
        data_u += dsize*n
        data_v += dsize*n
stop = time.time()
print('time: %.1e' % (stop - start))

start = time.time()
data_u = u.ctypes.data
data_v = v.ctypes.data
for i in range(k):
    ptr_ui = ctypes.c_void_p(data_u)
    ptr_vi = ctypes.c_void_p(data_v)
    for t in range(l):
        # much faster than the above apprently because data sits in cache
        s[i] = dot(mkl_n, ptr_ui, inc, ptr_vi, inc)
    data_u += dsize*n
    data_v += dsize*n
stop = time.time()
print('time: %.1e' % (stop - start))
