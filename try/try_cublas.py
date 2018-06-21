import ctypes
import numpy
import sys
sys.path.append('..')

POINTER = ctypes.POINTER

def shifted_ptr(ptr, shift):
    return ctypes.cast(ptr.value + shift, POINTER(ctypes.c_ubyte))

n = 1024 #*1024
v = numpy.ones((n,), dtype = numpy.float32)
v[0] = 1000
dev_v = POINTER(ctypes.c_ubyte)()
print(type(dev_v))

import raleigh.cuda.cuda as cuda

from raleigh.cuda.cublas import Cublas
from raleigh.cuda.cublas_algebra import Vectors

cublas = Cublas(numpy.float32)

print(cuda.malloc(ctypes.byref(dev_v), 0))

size = ctypes.c_int(4*n)
inc = ctypes.c_int(1)
print(cuda.malloc(ctypes.byref(dev_v), size))
ptr_v = ctypes.c_void_p(v.ctypes.data)
print(cuda.memcpy(dev_v, ptr_v, size, cuda.memcpyH2D))
#t = ctypes.c_float()
#print(cublas.norm(cublas.handle, ctypes.c_int(n), dev_v, inc, ctypes.byref(t)))
#r = t.value
#print(r)
s = ctypes.c_float()
print(cublas.dot(cublas.handle, ctypes.c_int(n), dev_v, inc, dev_v, inc, ctypes.byref(s)))
r = s.value
print(r)

#print(type(dev_v))

ptr_v = ctypes.cast(dev_v, ctypes.c_void_p)
s = ctypes.c_float()
ptr = shifted_ptr(ptr_v, 4)
#print(type(ptr))
#ptr = ctypes.cast(ptr_v.value + 4, POINTER(ctypes.c_ubyte))
print(cublas.dot(cublas.handle, ctypes.c_int(n - 1), ptr, inc, ptr, inc, ctypes.byref(s)))
r = s.value
print(r)

cuda.free(dev_v)

v = Vectors(n, 0)
