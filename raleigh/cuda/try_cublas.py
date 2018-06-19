import ctypes
import cuda
import numpy

from cublas import CUBLAS

POINTER = ctypes.POINTER

Cublas = CUBLAS(numpy.float32)

n = 1024*1024
v = numpy.ones((n,), dtype = numpy.float32)
dev_v = POINTER(ctypes.c_ubyte)()

size = ctypes.c_int(4*n)
print(cuda.malloc(ctypes.byref(dev_v), size))
print(cuda.memcpy(dev_v, ctypes.c_void_p(v.ctypes.data), size, cuda.memcpyH2D))
t = ctypes.c_float()
print(Cublas.norm(Cublas.handle, ctypes.c_int(n), dev_v, ctypes.c_int(1), ctypes.byref(t)))
r = t.value
print(r)

cuda.free(dev_v)