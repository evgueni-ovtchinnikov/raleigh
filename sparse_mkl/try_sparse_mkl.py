import numpy
import ctypes
import scipy.sparse as scsp
import time

n = 1024*1024
v = numpy.ones((n,), dtype = numpy.float32)

#dll = ctypes.cdll.LoadLibrary('./x64/Release/try_dll.dll')
dll = ctypes.CDLL('./sparse_mkl.dll', mode = ctypes.RTLD_GLOBAL)

numpy.random.seed(1) # make results reproducible
form = 'csr'
dt = numpy.float32

mA = 40000
nA = 30000
nnz = 200
den = nnz/min(mA, nA)
print('generating matrix of density %.2e...' % den)
A = scsp.random(mA, nA, den, form, dtype = dt)

maxk = 16
k = 6

ntests = 20

x = numpy.random.randn(maxk, nA).astype(dt)
y = numpy.zeros((maxk, mA), dtype = dt)
ptr_a = ctypes.c_void_p(A.data.ctypes.data)
indices = A.indices + 1
indptr = A.indptr + 1
ptr_ja  = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
ptr_ia  = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
ptr_u = ctypes.c_void_p(x.ctypes.data)
ptr_v = ctypes.c_void_p(y.ctypes.data)

print('testing...')
for k in range(1, 17):
    start = time.time()
    for t in range(ntests):
        dll.scsrmm(ctypes.c_int(mA), ctypes.c_int(nA), ctypes.c_int(k), \
                   ptr_ia, ptr_ja, ptr_a, ptr_u, ptr_v)
    stop = time.time()
    time_scsrmm = stop - start
    print('vectors: %d, time per vector: %.1e' % (k, time_scsrmm/k))
