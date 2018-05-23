from ctypes import *
import numpy

cudaMemcpyHostToHost          =   c_int(0)
cudaMemcpyHostToDevice        =   c_int(1)
cudaMemcpyDeviceToHost        =   c_int(2)
cudaMemcpyDeviceToDevice      =   c_int(3)

cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/bin'
cuda = CDLL(cuda_path + '/cudart64_70.dll', mode = RTLD_GLOBAL)
#cublas = CDLL('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0\\bin\\cublas64_70.dll', mode = RTLD_GLOBAL)
cublas = CDLL(cuda_path + '/cublas64_70.dll', mode = RTLD_GLOBAL)

cuda_malloc = cuda.cudaMalloc
cuda_malloc.argtypes = [POINTER(POINTER(c_ubyte)), c_int]
cuda_malloc.restype = c_int

cuda_free = cuda.cudaFree
cuda_free.restype = c_int

cuda_memcpy = cuda.cudaMemcpy
cuda_memcpy.restype = c_int

cublas_create = cublas.cublasCreate_v2
cublas_create.argtypes = [POINTER(POINTER(c_ubyte))]
cublas_create.restype = c_int

cublas_destroy = cublas.cublasDestroy_v2
#cublas_destroy.argtypes = [POINTER(c_ubyte)]
cublas_destroy.restype = c_int

handle = POINTER(c_ubyte)()
print(cublas_create(byref(handle)))

norm = cublas.cublasSnrm2_v2
norm.restype = c_int
n = 1024 #*1024
v = numpy.ones((n,), dtype = numpy.float32)
dev_v = POINTER(c_ubyte)()

size = c_int(4*n)
print(cuda_malloc(byref(dev_v), size))
print(cuda_memcpy(dev_v, c_void_p(v.ctypes.data), size, cudaMemcpyHostToDevice))
t = c_float()
print(norm(handle, c_int(n), dev_v, c_int(1), byref(t)))
r = t.value
print(r)
cuda_free(dev_v)

cublas_destroy(handle)
