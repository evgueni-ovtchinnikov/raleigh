# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
'''Interactive PCA demo.

Computes principal components of a dataset until stopped by the user.

Usage: interactive_pca <data_file> [gpu]

data_file    : the name of the file containing data
gpu          : run raleigh pca on GPU if this argument is present
'''

import numpy
import os
import sys
import timeit

from raleigh.interfaces.pca import pca, pca_error
from raleigh.core.solver import Options
from raleigh.interfaces.truncated_svd import UserStoppingCriteria


narg = len(sys.argv)
if narg < 2:
    usage = 'Usage: interactive_pca <data_file> [gpu]'
    raise SystemExit(usage)
data = numpy.load(sys.argv[1])
m = data.shape[0]
n = data.shape[1]
if len(data.shape) > 2:
    n = numpy.prod(data.shape[1:])
    data = numpy.reshape(data, (m, n))
arch = 'cpu' if narg < 3 else 'gpu!'

numpy.random.seed(1) # make results reproducible

opt = Options()
opt.stopping_criteria = UserStoppingCriteria(data, shift=True)
mean, trans, comps = pca(data, opt=opt, arch=arch)
ncomp = comps.shape[0]
print('%d principal components computed' % ncomp)
em, ef = pca_error(data, mean, trans, comps)
print('PCA error: max 2-norm %.1e, Frobenius norm %.1e' % (em, ef))

print('done')
