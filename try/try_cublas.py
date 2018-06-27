import ctypes
import numpy
import sys
sys.path.append('..')

POINTER = ctypes.POINTER

def conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a

#def shifted_ptr(ptr, shift):
#    return ctypes.cast(ptr.value + shift, POINTER(ctypes.c_ubyte))
def shifted_ptr(dev_ptr, shift):
    ptr = ctypes.cast(dev_ptr, ctypes.c_void_p)
    return ctypes.cast(ptr.value + shift, ctypes.POINTER(ctypes.c_ubyte))

n = 1024 #*1024
#v = numpy.ones((2, n), dtype = numpy.float32)
#v[0,0] = 1000
#dev_v = POINTER(ctypes.c_ubyte)()
#print(type(dev_v))
#
#import raleigh.cuda.cuda as cuda
#
from raleigh.cuda.cublas import Cublas
from raleigh.cuda.cublas_algebra import Vectors, Matrix
#
#cublas = Cublas(numpy.float32)
#
#print(cuda.malloc(ctypes.byref(dev_v), 0))
#print(cuda.free(dev_v))
#
#size = ctypes.c_int(4*n*2)
#inc = ctypes.c_int(1)
#print(cuda.malloc(ctypes.byref(dev_v), size))
#ptr_v = ctypes.c_void_p(v.ctypes.data)
#print(cuda.memcpy(dev_v, ptr_v, size, cuda.memcpyH2D))
##t = ctypes.c_float()
##print(cublas.norm(cublas.handle, ctypes.c_int(n), dev_v, inc, ctypes.byref(t)))
##r = t.value
##print(r)
#s = ctypes.c_float()
#print(cublas.dot(cublas.handle, ctypes.c_int(n), dev_v, inc, dev_v, inc, ctypes.byref(s)))
#r = s.value
#print(r)
#
##print(type(dev_v))
#
##ptr_v = ctypes.cast(dev_v, ctypes.c_void_p)
#s = ctypes.c_float()
#ptr = shifted_ptr(dev_v, 4*n)
##ptr = shifted_ptr(ptr_v, 4*n)
##ptr = shifted_ptr(ptr_v, 4)
##print(type(ptr))
##ptr = ctypes.cast(ptr_v.value + 4, POINTER(ctypes.c_ubyte))
##print(cublas.dot(cublas.handle, ctypes.c_int(n - 1), ptr, inc, ptr, inc, ctypes.byref(s)))
#print(cublas.dot(cublas.handle, ctypes.c_int(n), ptr, inc, ptr, inc, ctypes.byref(s)))
#r = s.value
#print(r)
#
#cuda.free(dev_v)

m = 3
#data_type = numpy.float32
data_type = numpy.complex64
cublas = Cublas(data_type)
z = numpy.ndarray((m, n), dtype = data_type)
for i in range(m):
    z[i, :] = i + 1
#z[0,:] = 1
#z[1,:] = 2
#print(z.dtype.type)
#print(z.itemsize)
#print(isinstance(z[0,0], complex))
##z[0,0] = 1000
#size = ctypes.c_int(2*n*z.itemsize)
#dev_z = POINTER(ctypes.c_ubyte)()
#print(cuda.malloc(ctypes.byref(dev_z), size))
#ptr_z = ctypes.c_void_p(z.ctypes.data)
#print(cuda.memcpy(dev_z, ptr_z, size, cuda.memcpyH2D))
#if data_type == numpy.float64 or data_type == numpy.complex128:
#    floats = ctypes.c_double * 2
#else:
#    floats = ctypes.c_float * 2
#s = floats()
#print(cublas.dot(cublas.handle, ctypes.c_int(n), dev_z, inc, dev_z, inc, s))
#sr = s[0]
#si = s[1]
#print(sr, si)

w = Vectors(z)
u = w.new_vectors(w.nvec())
v = w.new_vectors(w.nvec())
#u = Vectors(w)
#u = Vectors(n, w.nvec(), dtype = data_type)
ind = numpy.asarray([2, 1, 0])
w.copy(u, ind)

s = 2*numpy.ones((m,), dtype = data_type)
u.scale(s)
#w = Vectors(n, 0)

u.select(m - 1)

print(u.dimension())
print(u.nvec())
print(u.selected())
print(u.data_type())
print(u.is_complex())

s = u.dots(u)
print(s)

q = u.dot(w)
print(q)

v.select(m - 1)
u.copy(v)
s = v.dots(v)
print(s)

a = numpy.eye(n, dtype = data_type)
matrix = Matrix(a)
matrix.apply(u, v)
s = v.dots(v)
print(s)

w.multiply(q, v)
qq = v.dot(u)

print(qq)
print(numpy.dot(conjugate(q.T), q))

#print(q.flags)

#array2 = ctypes.c_float * 2
#t = array2()
#t[0] = 1
#t[1] = 4
