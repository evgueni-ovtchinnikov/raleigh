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
  -m <mim> , --mimgs=<mim>   number of images to add [default: 0]
  -e <err> , --imerr=<err>   acceptable image approximation error (<=0: to be 
                             decided interactively) [default: 0]
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
mi = int(args['--mimgs'])
max_err = float(args['--imerr'])
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
    def __init__(self, a, max_err = 0):
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.err = self.norms
        print('max data norm: %e' % numpy.amax(self.err))
        self.max_err = max_err
    def satisfied(self, solver):
        iterations = solver.iteration - self.iteration
        if solver.rcon <= self.ncon:
            return False
        now = time.time()
        elapsed_time = now - self.start_time
        self.elapsed_time += elapsed_time
        print('iterations: %d, eigenimages computed: %d, elapsed time %.2e' % \
            (solver.iteration, solver.rcon, self.elapsed_time))
#        print('elapsed time: +%.1e = %.2e, iterations: +%d = %d' % \
#            (elapsed_time, self.elapsed_time, iterations, solver.iteration))
        if self.max_err <= 0:
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
        self.ncon = solver.rcon
        if self.max_err > 0:
            done = err_max[1] <= self.max_err
        else:
            done = (input('more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        return done

numpy.random.seed(1) # make results reproducible

all_images = numpy.load(file)

m, ny, nx = all_images.shape
n = nx*ny

if ni < 0 or ni > m:
    ni = m

if ni < m:
    print('ising first %d images only...' % ni)
    m_all = m
    m = ni
    images = all_images[:m,:,:]
else:
    images = all_images

if ni + mi > m_all:
    mi = m_all - ni

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
opt.stopping_criteria = MyStoppingCriteria(images, max_err)

sigma, u, vt = partial_svd(images, opt, arch = arch)

ncon = sigma.shape[0]
iterations = opt.stopping_criteria.iteration
elapsed_time = opt.stopping_criteria.elapsed_time + \
    time.time() - opt.stopping_criteria.start_time
print('%d eigenimages computed in %d iterations, elapsed time: %.2e' % \
    (ncon, iterations, elapsed_time))

if mi > 0:
    print('adding %d images...' % mi)
    m = ni + mi
    images = all_images[:m,:,:]
    images = numpy.reshape(images, (m, n))
    opt.convergence_criteria.set_error_tolerance \
        ('residual eigenvector error', svec_tol)
    opt.stopping_criteria = MyStoppingCriteria(images, max_err)
    opt.block_size = ncon
    sigma, u, vt = partial_svd \
        (images, opt, nsv = ncon + mi, isv = vt.T, arch = arch)

ncon = sigma.shape[0]
iterations = opt.stopping_criteria.iteration
elapsed_time = opt.stopping_criteria.elapsed_time + \
    time.time() - opt.stopping_criteria.start_time
print('%d eigenimages computed in %d iterations, elapsed time: %.2e' % \
    (ncon, iterations, elapsed_time))

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
