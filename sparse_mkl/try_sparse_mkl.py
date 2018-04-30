import numpy
import ctypes
import scipy.sparse as scsp
import time

n = 1024*1024
v = numpy.ones((n,), dtype = numpy.float32)

#dll = ctypes.cdll.LoadLibrary('./x64/Release/try_dll.dll')
dll = ctypes.CDLL('./sparse_mkl.dll', mode = ctypes.RTLD_GLOBAL)

nthreads = dll.num_mkl_threads
nthreads.restype = ctypes.c_int
nt = nthreads()
print('max mkl threads: %d' % nt)
dll.set_num_mkl_threads(nt - 1)
nt = nthreads()
print('using %d mkl threads' % nt)

numpy.random.seed(1) # make results reproducible
form = 'csr'
dt = numpy.float64

mA = 40000
nA = 30000
nnz = 100
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
start = time.time()
for t in range(ntests):
    dll.dcsrmv(ctypes.c_int(mA), ptr_ia, ptr_ja, ptr_a, ptr_u, ptr_v)
stop = time.time()
time_csrmv = stop - start
print('matrix-vector time: %.1e' % time_csrmv)

for k in range(1, 17):
    start = time.time()
    for t in range(ntests):
        dll.dcsrmm(ctypes.c_int(mA), ctypes.c_int(nA), ctypes.c_int(k), \
                   ptr_ia, ptr_ja, ptr_a, ptr_u, ptr_v)
    stop = time.time()
    time_csrmm = stop - start
    print('vectors: %d, time per vector: %.1e' % (k, time_csrmm/k))
