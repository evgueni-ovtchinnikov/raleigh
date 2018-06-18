# -*- coding: utf-8 -*-
'''
Incremental PCA demo. Computes Principal Components for a set of 2D images
in small portions until stopped by user's entering 'n' in answer to 'more?'

Usage:
  try_ipca [--help | -h | options] <data>

Arguments:
  data .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -p <path>, --path=<path>   path to the data directory
                             [default: C:/Users/wps46139/Documents/Data/PCA]
  -b <blk> , --bsize=<blk>   block CG block size [default: 16]
  -t <tol> , --svtol=<tol>   singular vector error tolerance [default: 1e-2]

Created on Mon Jun 18 12:10:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

path = args['--path']
file = args['<data>']
svec_tol = float(args['--svtol'])
block_size = int(args['--bsize'])

import numpy
import sys
import time

sys.path.append('..')

from raleigh.solver import Options
from raleigh.ndarray.svd import partial_svd

class MyStoppingCriteria:
    def __init__(self):
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
    def satisfied(self, solver):
        iterations = solver.iteration - self.iteration
        if solver.rcon <= self.ncon:
            return False
        now = time.time()
        elapsed_time = now - self.start_time
        self.elapsed_time += elapsed_time
        print('elapsed time: +%.1e = %.2e, iterations: +%d = %d' % \
            (elapsed_time, self.elapsed_time, iterations, solver.iteration))
        lmd = solver.eigenvalues[self.ncon : solver.rcon]
        sigma = -numpy.sort(-numpy.sqrt(lmd))
        if self.ncon == 0:
            self.sigma = sigma[0]
        for i in range(solver.rcon - self.ncon):
            print('sigma[%4d] = %14f = %10f*sigma[0]' % \
            (self.ncon + i, sigma[i], sigma[i]/self.sigma))
        self.ncon = solver.rcon
        done = (input('more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        return done

numpy.random.seed(1) # make results reproducible

filepath = path + '/' + file
images = numpy.load(filepath)

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

m, ny, nx = images.shape
n = nx*ny

images = numpy.reshape(images, (m, n))

#block_size = 256

# set solver options
opt = Options()
opt.block_size = block_size
opt.max_iter = 300
#opt.verbosity = 1
opt.convergence_criteria.set_error_tolerance \
    ('kinematic eigenvector error', svec_tol)
opt.stopping_criteria = MyStoppingCriteria()

sigma, u, vt = partial_svd(images, opt)

iterations = opt.stopping_criteria.iteration
elapsed_time = opt.stopping_criteria.elapsed_time

print('iterations: %d, time: %.2e' % (iterations, elapsed_time))
