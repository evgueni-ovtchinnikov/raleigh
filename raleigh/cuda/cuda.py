# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:37:57 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes

cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/bin'
cuda = ctypes.CDLL(cuda_path + '/cudart64_70.dll', mode = ctypes.RTLD_GLOBAL)

POINTER = ctypes.POINTER

class CudaDeviceProp(ctypes.Structure):
    _fields_ = [ 
            ('name', ctypes.c_char * 256),
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
            ]

getDeviceCount = cuda.cudaGetDeviceCount
getDeviceProperties = cuda.cudaGetDeviceProperties
malloc = cuda.cudaMalloc
malloc.argtypes = [POINTER(POINTER(ctypes.c_ubyte)), ctypes.c_int]
malloc.restype = ctypes.c_int
free = cuda.cudaFree
free.restype = ctypes.c_int
memcpy = cuda.cudaMemcpy
memcpy.restype = ctypes.c_int
memcpyH2H = ctypes.c_int(0)
memcpyH2D = ctypes.c_int(1)
memcpyD2H = ctypes.c_int(2)
memcpyD2D = ctypes.c_int(3)



