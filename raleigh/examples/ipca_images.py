# -*- coding: utf-8 -*-
'''
Incremental PCA demo. Computes Principal Components for a set of 2D images
in small portions until stopped by user's entering 'n' in answer to 'more?'

Usage:
  ipca_images [--help | -h | options] <data>

Arguments:
  data  .npy file containing images as ndarray of dimensions (ni, ny, nx)

Options:
  -n <nim> , --nimgs=<nim>   number of images to use, negative=all [default: -1]
  -b <blk> , --bsize=<blk>   block CG block size [default: 64]
  -t <tol> , --svtol=<tol>   singular vector error tolerance [default: 1e-2]
  -a <arch>, --arch=<arch>   architecture [default: cpu]

Created on Mon Jun 18 12:10:20 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

#path = args['--path']
file = args['<data>']
ni = int(args['--nimgs'])
block_size = int(args['--bsize'])
svec_tol = float(args['--svtol'])
arch = args['--arch']

import math
import numpy
import sys
import time

sys.path.append('../..')

from raleigh.solver import Options
from raleigh.ndarray.svd import partial_svd, PSVDErrorCalculator

class MyStoppingCriteria:
    def __init__(self, a):
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.err = self.norms
        print('max data norm: %e' % numpy.amax(self.err))
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
        self.err = self.err_calc.update_errors()
        err_max = (numpy.amax(self.err), numpy.amax(self.err/self.norms))
        print('max err: abs %e, rel %e' % err_max)
        err_av = (numpy.sum(self.err)/len(self.err))
        print('average err: %e' % err_av)
#        std_dev = math.sqrt(self.err.var())
#        print('std dev: %e' % std_dev)
#        k = (len(self.err[self.err > err_av + 3*std_dev]))
#        print('above average + 3*std_dev: %d' % k)
        k = (len(self.err[self.err > 0.2]))
        print('above 0.2: %d' % k)
        self.ncon = solver.rcon
        done = (input('more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        return done

numpy.random.seed(1) # make results reproducible

images = numpy.load(file)

m, ny, nx = images.shape
n = nx*ny

if ni < 0:
    ni = m

if ni < m:
    print('ising first %d images only...' % ni)
    m = ni
    images = images[:m,:,:]

vmin = numpy.amin(images)
vmax = numpy.amax(images)
print('data range: %e to %e' % (vmin, vmax))

images = numpy.reshape(images, (m, n))

opt = Options()
opt.block_size = block_size
opt.max_iter = 300
opt.verbosity = -1
opt.convergence_criteria.set_error_tolerance \
    ('kinematic eigenvector error', svec_tol)
opt.stopping_criteria = MyStoppingCriteria(images)

sigma, u, vt = partial_svd(images, opt, arch = arch)

iterations = opt.stopping_criteria.iteration
elapsed_time = opt.stopping_criteria.elapsed_time + \
    time.time() - opt.stopping_criteria.start_time

print('iterations: %d, time: %.2e' % (iterations, elapsed_time))

ncon = sigma.shape[0]
#v = numpy.reshape(u.T, (ncon, m))
#u = numpy.reshape(vt, (ncon, ny, nx))    
coord = numpy.reshape((sigma*u).T, (ncon, m))
eigim = numpy.reshape(vt, (ncon, ny, nx))    
numpy.save('eigim.npy', eigim)
numpy.save('coord.npy', coord)
numpy.save('sigma.npy', sigma[:ncon])
#numpy.save('u.npy', u)
#numpy.save('v.npy', v)
print('done')
