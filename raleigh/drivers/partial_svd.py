# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Partial SVD of a matrix represented by a 2D ndarray.
"""

import math
import numpy
import numpy.linalg as nla
import time

try:
    input = raw_input
except NameError:
    pass

from ..core.solver import Problem, Solver, Options


''' For advanced users only.
'''
class PartialSVD:

    def __init__(self):
        self.sigma = None
        self.__left = None
        self.__right = None
        self.__mean = None
        self.__left_v = None
        self.__right_v = None
        self.__mean_v = None
        self.iterations = -1

    def compute(self, matrix, opt=Options(), nsv=(-1, -1), shift=False, \
                refine=True):
    
        op = matrix.as_operator()
        m, n = matrix.shape()
    
        transp = m < n
        if transp:
            n, m = m, n

        v = op.new_vectors(n)
#        v = matrix.as_vectors().new_vectors(0, n)
        dt = v.data_type()
        opSVD = _OperatorSVD(matrix, v, transp, shift)
        problem = Problem(v, opSVD)
        solver = Solver(problem)

        try:
            opt.stopping_criteria.err_calc.set_up(opSVD, solver, v, shift)
            if opt.verbosity > 0:
                print('partial SVD error calculation set up')
        except:
            if opt.verbosity > 0:
                print('partial SVD error calculation not requested')
            pass

        solver.solve(v, options=opt, which=nsv)
        if opt.verbosity > 0:
            print('operator application time: %.2e' % opSVD.time)

        nv = v.nvec()
        u = v.new_vectors(nv, m)
        if nv > 0:
            op.apply(v, u, transp)
            if shift:
                m, n = op.shape()
                dt = op.data_type()
                ones = numpy.ones((1, m), dtype=dt)
                e = v.new_vectors(1, m)
                e.fill(ones)
                w = v.new_vectors(1, n)
                op.apply(e, w, transp=True)
                w.scale(m*ones[0,:1])
                if not transp:
                    s = v.dot(w)
                    u.add(e, -1, s)
                else:
                    s = v.dot(e)
                    u.add(w, -1, s)
            if refine:
                sigma, q = u.svd()
                w = v.new_vectors(nv)
                v.multiply(_conj(q.T), w)
                w.copy(v)
##                u_data = u.data()
##                uu, sigma, uv = sla.svd(u_data.T, full_matrices=False)
##                u = u.new_vectors(uu.T)
##                wv = v.new_vectors(nv)
##                v.multiply(uv.T, wv)
##                wv.copy(v)
            else:
                sigma = numpy.sqrt(abs(u.dots(u)))
                u.scale(sigma)
                w = u.new_vectors(nv)
                ind = numpy.argsort(-sigma)
                sigma = sigma[ind]
                u.copy(w, ind)
                w.copy(u)
                w = v.new_vectors(nv)
                v.copy(w, ind)
                w.copy(v)
        else:
            sigma = numpy.ndarray((0,), dtype=v.data_type())
        self.sigma = sigma
        self.__mean_v = opSVD.mean_v()
        self.iterations = solver.iteration
        if transp:
            self.__left_v = v
            self.__right_v = u
        else:
            self.__left_v = u
            self.__right_v = v

    def mean(self):
        if self.__mean is None:
            if self.__mean_v is not None:
                self.__mean = self.__mean_v.data()
        return self.__mean

    def left(self):
        if self.__left is None:
            if self.__left_v is not None:
                self.__left = self.__left_v.data().T
        return self.__left

    def right(self):
        if self.__right is None:
            if self.__right_v is not None:
                self.__right = self.__right_v.data().T
        return self.__right

    def mean_v(self):
        return self.__mean_v

    def left_v(self):
        return self.__left_v

    def right_v(self):
        return self.__right_v


class AMatrix:

    def __init__(self, a, arch='cpu', copy_data=False):
        self.__arch = arch
        if arch[:3] == 'gpu':
            try:
                from ..algebra import cuda_wrap as cuda
                from ..algebra.dense_cublas import Matrix, Vectors
                self.__op = Matrix(a)
                self.__gpu = cuda
            except:
                if len(arch) > 3 and arch[3] == '!':
                    raise RuntimeError('cannot use GPU')
        else:
            from ..algebra.dense_cpu import Matrix, Vectors
#            from ..algebra.dense_numpy import Matrix, Vectors
            if copy_data:
                self.__op = Matrix(a.copy())
            else:
                self.__op = Matrix(a)
            self.__gpu = None
        self.__Vectors = Vectors
        self.__vectors = None
        vmin = numpy.amin(a)
        vmax = numpy.amax(a)
        self.__scale = max(abs(vmin), abs(vmax))

    def as_operator(self):
        return self.__op

    def as_vectors(self):
        if self.__vectors is None:
            self.__vectors = self.__Vectors(self.__op, shallow=True)
        return self.__vectors

    def arch(self):
        return self.__arch

    def gpu(self):
        return self.__gpu

    def dots(self):
        return self.__op.dots()

    def data_type(self):
        return self.__op.data_type()

    def shape(self):
        return self.__op.shape()

    def scale(self):
        return self.__scale


class PSVDErrorCalculator:
    def __init__(self, a):
        m, n = a.shape()
        self.dt = a.data_type()
        s = a.dots()
        self.norms = numpy.sqrt(s.reshape((m, 1)))
        self.solver = None
        self.err = None
        self.op = None
        self.m = m
        self.n = n
        self.shift = False
        self.ncon = 0
        self.err = self.norms.copy()
        self.aves = None
    def set_up(self, op, solver, eigenvectors, shift=False):
        self.op = op.op
        self.solver = solver
        self.eigenvectors = eigenvectors
        self.shift = shift
        if shift:
            self.ones = op.ones
            self.aves = op.aves
            s = self.aves.dots(self.aves)
            vb = eigenvectors.new_vectors(1, self.m)
            self.op.apply(self.aves, vb)
            b = vb.data().reshape((self.m, 1))
            t = (self.norms*self.norms).reshape((self.m, 1))
            x = t - 2*b + s*numpy.ones((self.m, 1))
            self.err = numpy.sqrt(abs(x))
        self.err_init = numpy.amax(self.err)
        self.err_init_f = nla.norm(self.err)
    def update_errors(self):
        ncon = self.eigenvectors.nvec()
        new = ncon - self.ncon
        if new > 0:
            err = self.err*self.err 
            x = self.eigenvectors
            sel = x.selected()
            x.select(new, self.ncon)
            m = self.m
            n = self.n
            if m < n:
                z = x.new_vectors(new, n)
                self.op.apply(x, z, transp=True)
                if self.shift:
                    s = x.dot(self.ones)
                    z.add(self.aves, -1, s)
                y = x.new_vectors(new, m)
                self.op.apply(z, y)
                if self.shift:
                    s = z.dot(self.aves)
                    y.add(self.ones, -1, s)
                q = x.dots(y, transp=True)
                q[q < 0] = 0
                err[q <= 0] = 0
            else:
                y = x.new_vectors(new, m)
                self.op.apply(x, y)
                if self.shift:
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                    # accurate orthogonalization needed!
                    s = y.dot(self.ones)
                    y.add(self.ones, -1.0/m, s)
                q = y.dots(y, transp=True)
            q = q.reshape((m, 1))
            err -= q
            err[err < 0] = 0
            self.err = numpy.sqrt(err)
            self.eigenvectors.select(sel[1], sel[0])
            self.ncon = ncon
        return self.err


class DefaultStoppingCriteria:

    def __init__(self, a, err_tol=0, norm='f', max_nsv=0, verb=0):
        self.shape = a.shape()
        self.scale = a.scale()
        self.err_tol = err_tol
        self.norm = norm
        self.max_nsv = max_nsv
        self.verb = verb
        self.ncon = 0
        self.sigma = 1
        self.iteration = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.err_calc = PSVDErrorCalculator(a)
        self.norms = self.err_calc.norms
        self.max_norm = numpy.amax(self.norms)
        self.f_norm = math.sqrt(numpy.sum(self.norms*self.norms))
        self.f = 0

    def satisfied(self, solver):
        self.norms = self.err_calc.norms
        m, n = self.shape
#        scale_max = self.scale*math.sqrt(n)
#        scale_f = self.scale*math.sqrt(m*n)
        scale_max = self.err_calc.err_init
        scale_f = self.err_calc.err_init_f
        if solver.rcon <= self.ncon:
            return False
        new = solver.rcon - self.ncon
        lmd = solver.eigenvalues[self.ncon : solver.rcon]
        sigma = -numpy.sort(-numpy.sqrt(abs(lmd)))
        if self.ncon == 0:
            self.sigma = sigma[0]
            self.err = self.err_calc.err
            self.f = numpy.sum(self.err*self.err)
        i = new - 1
        si = sigma[i]
        si_rel = si/self.sigma
        if self.norm == 'm':
            self.err = self.err_calc.update_errors()
            err_abs = numpy.amax(self.err)
            err_rel = err_abs/scale_max
#            err_rel = err_abs/self.max_norm
        elif self.norm == 'f':
            self.f -= numpy.sum(sigma*sigma)
            err_abs = math.sqrt(abs(self.f))
            err_rel = err_abs/scale_f
#            err_rel = err_abs/self.f_norm
        else:
            err_abs = si
            err_rel = si_rel
        now = time.time()
        elapsed_time = now - self.start_time
        self.elapsed_time += elapsed_time
        if self.norm in ['f', 'm']:
            msg = '%.2f sec: sigma[%d] = %.2e*sigma[0], truncation error = %.2e' % \
                  (self.elapsed_time, self.ncon + i, si_rel, err_rel)
        else:
            msg = '%.2f sec: sigma[%d] = %e = %.2e*sigma[0]' % \
                (self.elapsed_time, self.ncon + i, si, si_rel)
        self.ncon = solver.rcon
        done = False
        if self.err_tol != 0:
            if self.verb > 0:
                print(msg)
            if self.err_tol > 0:
                done = err_rel <= self.err_tol
            else:
                done = err_abs <= abs(self.err_tol)
        elif self.max_nsv < 1:
            done = (input(msg + ', more? ') == 'n')
        elif self.verb > 0:
            print(msg)
        self.iteration = solver.iteration
        self.start_time = time.time()
        done = done or self.max_nsv > 0 and self.ncon >= self.max_nsv
        return done


class _OperatorSVD:
    def __init__(self, matrix, v, transp=False, shift=False):
        self.op = matrix.as_operator()
        self.gpu = matrix.gpu()
        self.transp = transp
        self.shift = shift
        self.time = 0
        m, n = self.op.shape()
        if transp:
            self.w = v.new_vectors(0, n)
        else:
            self.w = v.new_vectors(0, m)
        if shift:
            dt = self.op.data_type()
            ones = numpy.ones((1, m), dtype=dt)
            self.ones = v.new_vectors(1, m)
            self.ones.fill(ones)
            self.aves = v.new_vectors(1, n)
            self.op.apply(self.ones, self.aves, transp=True)
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
            self.op.apply(x, z, transp=True)
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
            self.op.apply(z, y, transp=True)
        if self.gpu is not None:
            self.gpu.synchronize()
        stop = time.time()
        self.time += stop - start
    def mean(self):
        if self.shift:
            return self.aves.data()
        else:
            return None
    def mean_v(self):
        if self.shift:
            return self.aves
        else:
            return None


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def _conj(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a
