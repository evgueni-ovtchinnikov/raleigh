# -*- coding: utf-8 -*-
"""
Convenience function for Partial SVD of a ndarray via RALEIGH

Created on Wed Mar 21 14:06:26 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

import copy
import numpy
import numpy.linalg as nla
#import scipy
import sys
import time

raleigh_path = '../..'
if raleigh_path not in sys.path:
    sys.path.append(raleigh_path)

from raleigh.solver import Options, Problem, Solver

class OperatorSVD:
    def __init__(self, op, v, gpu, transp = False, shift = False):
        self.op = op
        self.gpu = gpu
        self.transp = transp
        self.shift = shift
        self.time = 0
        m, n = self.op.shape()
        if transp:
            self.w = v.new_vectors(0, n)
        else:
            self.w = v.new_vectors(0, m)
        if shift:
            dt = op.data_type()
            ones = numpy.ones((1, m), dtype = dt)
            self.ones = v.new_vectors(1, m)
            self.ones.fill(ones)
            self.aves = v.new_vectors(1, n)
            self.op.apply(self.ones, self.aves, transp = True)
            self.aves.scale(m*ones[0,:1])
    def apply(self, x, y):
        m, n = self.op.shape()
        k = x.nvec()
        start = time.time()
        if self.transp:
            if self.w.nvec() < k:
                self.w = x.new_vectors(k, n)
            z = self.w
            z.select(k)
            self.op.apply(x, z, transp = True)
            if self.shift:
                s = x.dot(self.ones)
                z.add(self.aves, -1, s)
            self.op.apply(z, y)
            if self.shift:
                s = z.dot(self.aves)
                y.add(self.ones, -1, s)
        else:
            if self.w.nvec() < k:
                self.w = x.new_vectors(k, m)
            z = self.w
            z.select(k)
            self.op.apply(x, z)
            if self.shift:
                s = z.dot(self.ones)
                z.add(self.ones, -1.0/m, s)
                # accurate orthogonalization needed!
                s = z.dot(self.ones)
                z.add(self.ones, -1.0/m, s)
            self.op.apply(z, y, transp = True)
        if self.gpu is not None:
            self.gpu.synchronize()
        stop = time.time()
        self.time += stop - start
    def mean(self):
        if self.shift:
            return self.aves
        else:
            return None

class PSVDErrorCalculator:
    def __init__(self, a):
#        self.reset()
#    def reset(self):
        self.solver = None
        self.err = None
        self.op = None
        m, n = a.shape
        self.m = m
        self.n = n
        self.dt = a.dtype.type
        self.shift = False
        self.ncon = 0
        self.norms = nla.norm(a, axis = 1).reshape((m, 1))
        self.err = self.norms.copy()
    def set_up(self, op, solver, eigenvectors, shift = False):
        self.op = op
        self.solver = solver
        self.eigenvectors = eigenvectors
        self.shift = shift
        if shift:
            self.ones = eigenvectors.new_vectors(1, self.m)
            self.ones.fill(1.0)
            self.aves = eigenvectors.new_vectors(1, self.n)
            self.op.apply(self.ones, self.aves, transp = True)
            self.aves.scale(self.m*numpy.ones((1,)))
            s = self.aves.dots(self.aves)
            vb = eigenvectors.new_vectors(1, self.m)
            self.op.apply(self.aves, vb)
            b = vb.data().reshape((self.m, 1))
            t = (self.norms*self.norms).reshape((self.m, 1))
            x = t - 2*b + s*numpy.ones((self.m, 1))
            self.err = numpy.sqrt(abs(x))
    def update_errors(self):
        ncon = self.eigenvectors.nvec()
        new = ncon - self.ncon
        if new > 0:
            x = self.eigenvectors
            sel = x.selected()
            x.select(new, self.ncon)
            m = self.m
            n = self.n
            if m < n:
                z = x.new_vectors(new, n)
                self.op.apply(x, z, transp = True)
                if self.shift:
                    s = x.dot(self.ones)
                    z.add(self.aves, -1, s)
                y = x.new_vectors(new, m)
                self.op.apply(z, y)
                if self.shift:
                    s = z.dot(self.aves)
                    y.add(self.ones, -1, s)
                q = x.dots(y, transp = True)
            else:
                y = x.new_vectors(new, m)
                self.op.apply(x, y)
                if self.shift:
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                    # accurate orthogonalization needed!
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                q = y.dots(y, transp = True)
            s = self.err*self.err - q.reshape((m, 1))
            s[s < 0] = 0
            self.err = numpy.sqrt(s)
            self.eigenvectors.select(sel[1], sel[0])
            self.ncon = ncon
        return self.err

class DefaultConvergenceCriteria:
    def __init__(self):
        self.tolerance = 1e-3
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('res', i)
        res = solver.res[i]
        lmd = solver.lmd[i]
        return err >= 0 and err <= self.tolerance and res <= 0.25*abs(lmd)
        
class DefaultStoppingCriteria:
    def __init__(self, a, err_tol = 0, sigma_th = 0, max_nsv = 0):
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.err = self.norms
        #print('max data norm: %e' % numpy.amax(self.err))
        self.err_tol = err_tol
        self.sigma_th = sigma_th
        self.max_nsv = max_nsv
    def satisfied(self, solver):
        self.norms = self.err_calc.norms
        if solver.rcon <= self.ncon:
            return False
        self.err = self.err_calc.update_errors()
        err_rel = numpy.amax(self.err/self.norms)
        lmd = solver.eigenvalues[self.ncon : solver.rcon]
        sigma = -numpy.sort(-numpy.sqrt(abs(lmd)))
        if self.ncon == 0:
            self.sigma = sigma[0]
        now = time.time()
        new = solver.rcon - self.ncon
        elapsed_time = now - self.start_time
        self.elapsed_time += elapsed_time
        i = new - 1
        si = sigma[i]
        si_rel = si/self.sigma
        if self.err_tol <= 0:
            msg = '%.2f sec: sigma[%d] = %e = %.2e*sigma[0], err = %.2e' % \
                (self.elapsed_time, self.ncon + i, si, si_rel, err_rel)
        else:
            print('%.2f sec: sigma[%d] = %.2e*sigma[0], truncation error = %.2e' % \
                  (self.elapsed_time, self.ncon + i, si_rel, err_rel))
        self.ncon = solver.rcon
        if self.err_tol > 0 or self.sigma_th > 0:
            done = err_rel <= self.err_tol or si_rel <= self.sigma_th
        else:
            done = (input(msg + ', more? ') == 'n')
        self.iteration = solver.iteration
        self.start_time = time.time()
        done = done or self.max_nsv > 0 and self.ncon >= self.max_nsv
        return done

def conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a

class PartialSVD:
    def __init__(self):
        self.sigma = None
        self.u = None
        self.v = None
        self.mean = None
        self.iterations = -1
    def compute(self, a, opt, nsv = (-1, -1), isv = (None, None), \
                shift = False, arch = 'cpu'):
    
        if arch[:3] == 'gpu':
            try:
                import raleigh.cuda.cuda as cuda
                from raleigh.cuda.cublas_algebra import Vectors, Matrix
                op = Matrix(a)
                gpu = cuda
            except:
                if len(arch) > 3 and arch[3] == '!':
                    raise RuntimeError('cannot use GPU')
                gpu = None
        else:
            gpu = None
        if gpu is None:
            from raleigh.algebra import Vectors, Matrix
    #        from raleigh.ndarray.numpy_algebra import Vectors, Matrix
            op = Matrix(a)
    
        m, n = a.shape
        dt = a.dtype.type
    
        isvec = ()
        for i in range(2):
            if isv[i] is not None:
                k, l = isv[i].shape
                if k != n:
                    msg = 'initial singular vectors must have dimension %d, not %d'
                    raise ValueError(msg % (n, k))
                isvec += (Vectors(isv[i].T),)
            else:
                isvec += (None,)
        isv = isvec
    
        transp = m < n
        if transp:
            n, m = m, n
            isvec = ()
            for i in range(2):
                if isv[i] is not None:
                    tmp = Vectors(n, l, data_type = dt)
                    op.apply(isv[i], tmp)
                    isvec += (tmp,)
                else:
                    isvec += (None,)
            isv = isvec
    
        v = Vectors(n, data_type = dt)
        opSVD = OperatorSVD(op, v, gpu, transp, shift)
        problem = Problem(v, lambda x, y: opSVD.apply(x, y))
        solver = Solver(problem)
    
        try:
            opt.stopping_criteria.err_calc.set_up(op, solver, v, shift)
            if opt.verbosity > 0:
                print('partial SVD error calculation set up')
        except:
            if opt.verbosity > 0:
                print('partial SVD error calculation not requested')
            pass
    
        solver.solve(v, opt, which = nsv, init = isv)
        if opt.verbosity > 0:
            print('operator application time: %.2e' % opSVD.time)
    
        nv = v.nvec()
        u = Vectors(m, nv, v.data_type())
        if nv > 0:
            op.apply(v, u, transp)
            if shift:
                m, n = a.shape
                dt = op.data_type()
                ones = numpy.ones((1, m), dtype = dt)
                e = Vectors(m, 1, data_type = dt)
                e.fill(ones)
                w = Vectors(n, 1, data_type = dt)
                op.apply(e, w, transp = True)
                w.scale(m*ones[0,:1])
                if not transp:
                    s = v.dot(w)
                    u.add(e, -1, s)
                else:
                    s = v.dot(e)
                    u.add(w, -1, s)
    
    #        vv = v.dot(v)
    #        if nsv[0] == 0:
    #            uu = -u.dot(u)
    #        else:
    #            uu = u.dot(u)
    #        lmd, x = scipy.linalg.eigh(uu, vv) #, turbo = False)
    #        w = v.new_vectors(nv)
    #        v.multiply(x, w)
    #        w.copy(v)
    #        w = u.new_vectors(nv)
    #        u.multiply(x, w)
    #        w.copy(u)
            sigma = numpy.sqrt(abs(u.dots(u)))
            u.scale(sigma)
            ind = numpy.argsort(-sigma)
            sigma = sigma[ind]
            u = u.data().T[:, ind]
            v = v.data().T[:, ind]
        else:
            sigma = numpy.ndarray((0,), dtype = v.data_type())
            u = None
            v = None
        self.sigma = sigma
        self.mean = opSVD.mean()
        self.iterations = solver.iteration
        if transp:
            self.u = v
            self.v = u
            return sigma, v, conj(u.T)
    #        return sigma, v.data().T, conj(u.data())
        else:
            self.u = u
            self.v = v
            return sigma, u, conj(v.T)
    #        return sigma, u.data().T, conj(v.data())

class TSVD:
    def __init__(self):
        self.sigma = None
        self.u = None
        self.v = None
        self.iterations = -1
    def compute(self, a, opt, nsv = -1, tol = 0, th = 0, msv = 0, \
                  isv = None, shift = False, \
                  arch = 'cpu'):
        if opt.convergence_criteria is None:
            opt = copy.deepcopy(opt)
            opt.convergence_criteria = DefaultConvergenceCriteria()
        if opt.stopping_criteria is None and nsv < 0:
            opt = copy.deepcopy(opt)
            opt.stopping_criteria = DefaultStoppingCriteria(a, tol, th, msv)
        psvd = PartialSVD()
        psvd.compute(a, opt, (0, nsv), (None, isv), shift, arch)
        self.sigma = psvd.sigma
        self.u = psvd.u
        self.v = psvd.v
        self.mean = psvd.mean
        self.iterations = psvd.iterations
        return self.sigma, self.u, self.v.T

def truncated_svd(a, opt, nsv = -1, tol = 0, th = 0, msv = 0, \
                  isv = None, shift = False, \
                  arch = 'cpu'):
    tsvd = TSVD()
    return tsvd.compute(a, opt = opt, nsv = nsv, tol = tol, th = th, \
                        msv = msv, isv = isv, shift = shift, arch = arch)
#    if opt.convergence_criteria is None:
#        opt = copy.deepcopy(opt)
#        opt.convergence_criteria = DefaultConvergenceCriteria()
#    if opt.stopping_criteria is None and nsv < 0:
#        opt = copy.deepcopy(opt)
#        opt.stopping_criteria = DefaultStoppingCriteria(a, tol, th, msv)
#    psvd = PartialSVD()
#    return psvd.compute(a, opt, (0, nsv), (None, isv), shift, arch)

def pca(a, opt = Options(), npc = -1, tol = 0, th = 0, msv = 0, ipc = None, \
        arch = 'cpu'):
    m, n = a.shape
    opt = copy.deepcopy(opt)
    block_size = opt.block_size
    if block_size < 1 and npc < 0:
        b = max(1, min(m, n)//100)
        block_size = 32
        while block_size <= b - 16:
            block_size += 32
        #print('using block size %d' % block_size)
        opt.block_size = block_size
    max_iter = opt.max_iter
    if max_iter < 0:
        opt.max_iter = max(100, min(m, n))
        #print('max_iter = %d' % opt.max_iter)
    return truncated_svd(a, opt = opt, nsv = npc, tol = tol, th = th, \
        msv = msv, isv = ipc, shift = True, arch = arch)
