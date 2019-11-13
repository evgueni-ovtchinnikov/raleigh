# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""CUDA Toolkit loader/wrapper.
"""

import ctypes
import glob
import os
from sys import platform

from . import verbosity


def _find_cuda_path(path):
    n = len(path)
    i = 0
    while i < n:
        cpath = path[i:]
        j = cpath.find('CUDA')
        if j < 0:
            return None
        f = cpath[: j].rfind(';')
        if f < 0:
            f = 0
        else:
            f += 1
        t = cpath[j :].find(';')
        if t < 0:
            t = len(cpath)
        else:
            t += j
        b = cpath[f : t].find('bin')
        if b < 0:
            i += t
            continue
        return cpath[f : t]
    return None


try:
    if platform == 'win32':
        cuda_path = _find_cuda_path(os.environ['PATH'])
        cudart_dll = glob.glob(cuda_path + '/cudart64*')[0]
        cuda = ctypes.CDLL(cudart_dll, mode=ctypes.RTLD_GLOBAL)
    else:
        cuda = ctypes.CDLL('libcudart.so', mode=ctypes.RTLD_GLOBAL)

    v = ctypes.c_int()
    cuda.cudaRuntimeGetVersion(ctypes.byref(v))
    version = v.value
    if verbosity.level > 0:
        print('CUDA version: %d' % version)
except:
    if verbosity.level > 0:
        print('CUDA Toolkit not found, switching to cpu...')
    raise RuntimeError('CUDA Toolkit not found')


if version >= 10000:
    UUID_SIZE = 32
else:
    UUID_SIZE = 0

POINTER = ctypes.POINTER


class CudaDeviceProp(ctypes.Structure):
    _fields_ = [ 
            ('name', ctypes.c_char * 256),
            ('uuid', ctypes.c_char * UUID_SIZE),
            ('totalGlobalMem', ctypes.c_size_t),
            ('sharedMemPerBlock', ctypes.c_size_t),
            ('regsPerBlock', ctypes.c_int),
            ('warpSize', ctypes.c_int),
            ('memPitch', ctypes.c_size_t),
            ('maxThreadsPerBlock', ctypes.c_int),
            ('maxThreadsDim', ctypes.c_int * 3), 
            ('maxGridSize', ctypes.c_int * 3), 
            ('clockRate', ctypes.c_int),
            ('totalConstMem', ctypes.c_size_t),
            ('major', ctypes.c_int),
            ('minor', ctypes.c_int),
            ('textureAlignment', ctypes.c_size_t),
            ('texturePitchAlignment', ctypes.c_size_t),
            ('deviceOverlap', ctypes.c_int),
            ('multiProcessorCount', ctypes.c_int),
            ('kernelExecTimeoutEnabled', ctypes.c_int),
            ('integrated', ctypes.c_int),
            ('canMapHostMemory', ctypes.c_int),
            ('computeMode', ctypes.c_int),
            ('maxTexture1D', ctypes.c_int),
            ('maxTexture1DMipmap', ctypes.c_int),
            ('maxTexture1DLinear', ctypes.c_int),
            ('maxTexture2D', ctypes.c_int * 2), 
            ('maxTexture2DMipmap', ctypes.c_int * 2), 
            ('maxTexture2DLinear', ctypes.c_int * 3), 
            ('maxTexture2DGather', ctypes.c_int * 2), 
            ('maxTexture3D', ctypes.c_int * 3), 
            ('maxTexture3DAlt', ctypes.c_int * 3), 
            ('maxTextureCubemap', ctypes.c_int),
            ('maxTexture1DLayered', ctypes.c_int * 2), 
            ('maxTexture2DLayered', ctypes.c_int * 3), 
            ('maxTextureCubemapLayered', ctypes.c_int * 2), 
            ('maxSurface1D', ctypes.c_int),
            ('maxSurface2D', ctypes.c_int * 2), 
            ('maxSurface3D', ctypes.c_int * 3), 
            ('maxSurface1DLayered', ctypes.c_int * 2), 
            ('maxSurface2DLayered', ctypes.c_int * 3), 
            ('maxSurfaceCubemap', ctypes.c_int),
            ('maxSurfaceCubemapLayered', ctypes.c_int * 2), 
            ('surfaceAlignment', ctypes.c_size_t),
            ('concurrentKernels', ctypes.c_int),
            ('ECCEnabled', ctypes.c_int),
            ('pciBusID', ctypes.c_int),
            ('pciDeviceID', ctypes.c_int),
            ('pciDomainID', ctypes.c_int),
            ('tccDriver', ctypes.c_int),
            ('asyncEngineCount', ctypes.c_int),
            ('unifiedAddressing', ctypes.c_int),
            ('memoryClockRate', ctypes.c_int),
            ('memoryBusWidth', ctypes.c_int),
            ('l2CacheSize', ctypes.c_int),
            ('maxThreadsPerMultiProcessor', ctypes.c_int),
            ('streamPrioritiesSupported', ctypes.c_int),
            ('globalL1CacheSupported', ctypes.c_int),
            ('localL1CacheSupported', ctypes.c_int),
            ('sharedMemPerMultiprocessor', ctypes.c_size_t),
            ('regsPerMultiprocessor', ctypes.c_int),
            ('managedMemSupported', ctypes.c_int),
            ('isMultiGpuBoard', ctypes.c_int),
            ('multiGpuBoardGroupID', ctypes.c_int),
            ('singleToDoublePrecisionPerfRatio', ctypes.c_int),
            ('pageableMemoryAccess', ctypes.c_int),
            ('concurrentManagedAccess', ctypes.c_int),
            ('rest', ctypes.c_int * 1024)
            ]


getDeviceCount = cuda.cudaGetDeviceCount
getDeviceProperties = cuda.cudaGetDeviceProperties
synchronize = cuda.cudaDeviceSynchronize
malloc = cuda.cudaMalloc
malloc.argtypes = [POINTER(POINTER(ctypes.c_ubyte)), ctypes.c_int]
malloc.restype = ctypes.c_int
free = cuda.cudaFree
free.argtypes = [ctypes.c_void_p]
free.restype = ctypes.c_int
memset = cuda.cudaMemset
memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
memset.restype = ctypes.c_int
memcpy = cuda.cudaMemcpy
memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
memcpy.restype = ctypes.c_int
memcpy2D = cuda.cudaMemcpy2D
memcpy2D.restype = ctypes.c_int
memcpyH2H = ctypes.c_int(0)
memcpyH2D = ctypes.c_int(1)
memcpyD2H = ctypes.c_int(2)
memcpyD2D = ctypes.c_int(3)
createStream = cuda.cudaStreamCreate
createStream.restype = ctypes.c_int
deleteStream = cuda.cudaStreamDestroy

numDevices = ctypes.c_int()
getDeviceCount(ctypes.byref(numDevices))
if verbosity.level > 1:
    print('devices found: %d' % numDevices.value)
    for x in range(numDevices.value):
        devProp = CudaDeviceProp()
        getDeviceProperties(ctypes.byref(devProp), x)
        print('device: %s' % devProp.name)
        print('memory: %d' % devProp.totalGlobalMem)
        print('multiprocessors: %d' % devProp.multiProcessorCount)
        print('threads per mp: %d' % devProp.maxThreadsPerMultiProcessor)
        print('registers per block: %d'% devProp.regsPerBlock)
        print('shared memory per block: %d' % devProp.sharedMemPerBlock)
