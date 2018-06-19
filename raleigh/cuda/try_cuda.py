# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:20:59 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import ctypes
import cuda

numDevices = ctypes.c_int()
cuda.getDeviceCount(ctypes.byref(numDevices))
print('devices found: %d' % numDevices.value)

for x in range(numDevices.value):
    devProp = cuda.CudaDeviceProp()
    cuda.getDeviceProperties(ctypes.byref(devProp), x)
    print('device: %s' % devProp.name)
    print('memory: %d' % devProp.totalGlobalMem)
    print('multiprocessors: %d' % devProp.multiProcessorCount)
    print('threads per mp: %d' % devProp.maxThreadsPerMultiProcessor)
    print('registers per block: %d'% devProp.regsPerBlock)
    print('shared memory per block: %d' % devProp.sharedMemPerBlock)
