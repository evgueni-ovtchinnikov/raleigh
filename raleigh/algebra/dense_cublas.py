# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''GPU implementation of RALEIGH dense algebra.
'''

import ctypes
import numbers
import numpy
import scipy

from . import cuda_wrap as cuda
from . import cublas_handle as cublas
from .cublas_wrap import Cublas


class Vectors:
    '''CUBLAS implementation of Vectors type.
    '''

    '''========== Methods required by RALEIGH core solver ==========
    '''

    def new_vectors(self, arg=0, dim=None):
        if isinstance(arg, numbers.Number):
            nv = int(arg)
            if dim is None:
                dim = self.dimension()
            return Vectors(dim, nv, self.data_type()) #, cublas=self.__cublas)
        else:
            return Vectors(arg) #, cublas=self.__cublas)

    def clone(self):
        return Vectors(self)

    def append(self, other, axis=0):
        if other.nvec() < 1:
            return
        inc = ctypes.c_int(1)
        n = self.__vdim
        vsize = self.__dsize * n
        if axis == 1:
            m, n = self.shape()
            l, n_other = other.shape()
            if m != l:
                msg = 'Cannot append %d vectors to %d vectors' % (l, m)
                raise ValueError(msg)
            type_self = self.data_type()
            type_other = other.data_type()
            if type_self != type_other:
                rs = repr(type_self)
                ro = repr(type_other)
                msg = 'Cannot append %s vectors to %s vectors' % (ro, rs)
                raise ValueError(msg)
            vsize_other = self.__dsize * n_other
            n_new = n + n_other
            vsize_new = vsize + vsize_other
            vdata = _Data(m * vsize_new)
            ptr_n = vdata.data_ptr()
            ptr_s = self.data_ptr()
            ptr_o = other.data_ptr()
            ldn = ctypes.c_int(vsize_new)
            lds = ctypes.c_int(vsize)
            ldo = ctypes.c_int(vsize_other)
            h = ctypes.c_int(m)
            _try_calling(cuda.memcpy2D(ptr_n, ldn, ptr_s, lds, lds, h, \
                         cuda.memcpyD2D))
            ptr = _shifted_ptr(ptr_n, vsize)
            _try_calling(cuda.memcpy2D(ptr, ldn, ptr_o, ldo, ldo, h, \
                         cuda.memcpyD2D))
            self.__vdata = vdata
            self.__vdim = n_new
            return
        i, m = self.selected()
        j, l = other.selected()
        nvec = i + m + l
        if nvec > self.__mvec:
            mvec = ((nvec - 1)//self.__min_inc + 1)*self.__min_inc
#            mvec = ((nvec - 1)//Vectors.MIN_INC + 1)*Vectors.MIN_INC
#            print('allocating %d vectors...' % mvec)
            size = mvec * vsize
            vdata = _Data(size)
            data = vdata.data_ptr()
            if i + m > 0:
                mn = ctypes.c_int((i + m)*n)
                ptr_u = self.all_data_ptr()
                self.__cublas.copy(self.__cublas.handle, mn, ptr_u, inc, data, \
                                   inc)
            self.__vdata = vdata
            self.__mvec = mvec
            if self.__min_inc < self.__max_inc:
                self.__min_inc *= 2
        data = self.all_data_ptr()
        data_v = other.all_data_ptr()
        ptr_v = _shifted_ptr(data_v, j*vsize)
        ptr = _shifted_ptr(data, int(i + m)*vsize)
        ln = ctypes.c_int(l*n)
#        print('copying %d vectors...' % l)
        self.__cublas.copy(self.__cublas.handle, ln, ptr_v, inc, ptr, inc)
        self.__nvec = nvec
        self.select_all()

    def dimension(self):
        return self.__vdim

    def nvec(self):
        return self.__selected[1]

    def select(self, nv, first=0):
        assert nv <= self.__nvec and first >= 0
        self.__selected = (first, nv)

    def selected(self):
        return self.__selected

    def data_type(self):
        return self.__dtype

    def fill_random(self):
        n = self.dimension()
        m = self.nvec()
        if m < 1:
            return
        data = numpy.random.rand(m, n).astype(self.data_type())
        data *= 2
        data -= 1
        vsize = n * self.__dsize
        size = m * vsize
        ptr_u = self.data_ptr()
        ptr_v = ctypes.c_void_p(data.ctypes.data)
        _try_calling(cuda.memcpy(ptr_u, ptr_v, size, cuda.memcpyH2D))

    def copy(self, other, ind=None):
        i, m = self.selected()
        j, l = other.selected()
        vdim = self.dimension()
        inc = ctypes.c_int(1)
        vsize = self.__cublas.dsize * vdim
        data_u = self.all_data_ptr()
        data_v = other.all_data_ptr()
        if ind is None:
            n = ctypes.c_int(m*vdim)
            ptr_u = _shifted_ptr(data_u, i*vsize)
            ptr_v = _shifted_ptr(data_v, j*vsize)
            self.__cublas.copy(self.__cublas.handle, n, ptr_u, inc, ptr_v, inc)
        else:
            n = ctypes.c_int(vdim)
            l = len(ind)
            for k in range(l):
                ptr_u = _shifted_ptr(data_u, int(ind[k])*vsize)
                ptr_v = _shifted_ptr(data_v, (j + k)*vsize)
                self.__cublas.copy \
                    (self.__cublas.handle, n, ptr_u, inc, ptr_v, inc)

    def scale(self, s, multiply=False):
        f, m = self.selected()
        vdim = self.dimension()
        n = ctypes.c_int(vdim)
        inc = ctypes.c_int(1)
        vsize = self.__cublas.dsize * vdim
        data_u = self.all_data_ptr()
        if multiply:
            for i in range(m):
                ptr_u = _shifted_ptr(data_u, (f + i)*vsize)
                r = self.__to_floats(s[i])
                self.__cublas.scal(self.__cublas.handle, n, r, ptr_u, inc)
        else:
            for i in range(m):
                ptr_u = _shifted_ptr(data_u, (f + i)*vsize)
                if s[i] != 0.0:
                    r = self.__to_floats(1.0/s[i])
                    self.__cublas.scal(self.__cublas.handle, n, r, ptr_u, inc)

    def dots(self, other, transp=False):
        if transp:
            m = self.nvec()
            n = self.dimension()
            c_n = ctypes.c_int(n)
            c_m = ctypes.c_int(m)
            c_k = ctypes.c_int(1)
            d_pu = self.data_ptr()
            d_pv = other.data_ptr()
            size_dptr = ctypes.sizeof(d_pu)
            alpha = self.__to_floats(1.0)
            beta = self.__to_floats(0.0)
            size_w = n*self.__dsize
            w = numpy.ndarray((n,), dtype=self.data_type())
            h_pw = ctypes.c_void_p(w.ctypes.data)
            d_pw = Cublas.CtypesPtr()
            _try_calling(cuda.malloc(ctypes.byref(d_pw), size_w))
            iota = numpy.arange(n)*self.__dsize
            ptr_u = ctypes.cast(d_pu, ctypes.c_void_p)
            ptr_v = ctypes.cast(d_pv, ctypes.c_void_p)
            ptr_w = ctypes.cast(d_pw, ctypes.c_void_p)
            h_ppu = iota + ptr_u.value
            h_ppv = iota + ptr_v.value
            h_ppw = iota + ptr_w.value
            size_pp = n*size_dptr
            d_ppu = Cublas.CtypesPtr()
            d_ppv = Cublas.CtypesPtr()
            d_ppw = Cublas.CtypesPtr()
            _try_calling(cuda.malloc(ctypes.byref(d_ppu), size_pp))
            _try_calling(cuda.malloc(ctypes.byref(d_ppv), size_pp))
            _try_calling(cuda.malloc(ctypes.byref(d_ppw), size_pp))
            _try_calling(cuda.memcpy(d_ppu, h_ppu.ctypes.data, size_pp, cuda.memcpyH2D))
            _try_calling(cuda.memcpy(d_ppv, h_ppv.ctypes.data, size_pp, cuda.memcpyH2D))
            _try_calling(cuda.memcpy(d_ppw, h_ppw.ctypes.data, size_pp, cuda.memcpyH2D))
            if self.__is_complex:
                Trans = Cublas.ConjTrans
            else:
                Trans = Cublas.Trans
            _try_calling(self.__cublas.gemm_batched(self.__cublas.handle, \
                Cublas.NoTrans, Trans, \
                c_k, c_k, c_m, alpha, d_ppu, c_n, d_ppv, c_n, \
                beta, d_ppw, c_k, c_n))
            _try_calling(cuda.memcpy(h_pw, d_pw, size_w, cuda.memcpyD2H))
            _try_calling(cuda.free(d_ppu))
            _try_calling(cuda.free(d_ppv))
            _try_calling(cuda.free(d_ppw))
            _try_calling(cuda.free(d_pw))
            return w
        iu = self.first()
        iv = other.first()
        vdim = self.dimension()
        dsize = self.__dsize
        vsize = dsize * vdim
        m = self.nvec()
        n = ctypes.c_int(vdim)
        inc = ctypes.c_int(1)
        data_u = self.all_data_ptr()
        data_v = other.all_data_ptr()
        w = numpy.ndarray((m,), dtype=self.data_type())
        for i in range(m):
            ptr_u = _shifted_ptr(data_u, (iu + i)*vsize)
            ptr_v = _shifted_ptr(data_v, (iv + i)*vsize)
            s = self.__floats()
            self.__cublas.dot \
                (self.__cublas.handle, n, ptr_v, inc, ptr_u, inc, s)
            if self.is_complex():
                w[i] = s[0] + 1j * s[1]
            else:
                w[i] = s[0]
        return w

    def dot(self, other):
        m = self.nvec()
        n = self.dimension()
        k = other.nvec()
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        c_k = ctypes.c_int(k)
        dptr_u = other.data_ptr()
        dptr_v = self.data_ptr()
        dptr_q = ctypes.POINTER(ctypes.c_ubyte)()
        size_q = int(k*m*self.__dsize)
        _try_calling(cuda.malloc(ctypes.byref(dptr_q), size_q))
        if self.is_complex():
            Trans = Cublas.ConjTrans
        else:
            Trans = Cublas.Trans
        one = self.__to_floats(1.0)
        zero = self.__to_floats(0.0)
        self.__cublas.gemm(self.__cublas.handle, Trans, Cublas.NoTrans, \
            c_m, c_k, c_n, one, dptr_v, c_n, dptr_u, c_n, zero, dptr_q, c_m)
        q = numpy.ndarray((k, m), dtype=self.data_type())
        hptr_q = ctypes.c_void_p(q.ctypes.data)
        _try_calling(cuda.memcpy(hptr_q, dptr_q, size_q, cuda.memcpyD2H))
        _try_calling(cuda.free(dptr_q))
        return _conjugate(q)

    def multiply(self, a, output):
        m = a.shape[1]
        n = self.dimension()
        k = self.nvec()
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        c_k = ctypes.c_int(k)
        dptr_u = output.data_ptr()
        dptr_v = self.data_ptr()
        dptr_q = ctypes.POINTER(ctypes.c_ubyte)()
        size_q = int(k*m*self.__dsize)
        _try_calling(cuda.malloc(ctypes.byref(dptr_q), size_q))
        if a.flags['C_CONTIGUOUS'] or a.flags['F_CONTIGUOUS']:
            q = a
        else:
            q = a.copy()
        hptr_q = ctypes.c_void_p(q.ctypes.data)
        _try_calling(cuda.memcpy(dptr_q, hptr_q, size_q, cuda.memcpyH2D))
        if q.flags['C_CONTIGUOUS']:
            Trans = Cublas.Trans
            ldq = c_m
        elif q.flags['F_CONTIGUOUS']:
            Trans = Cublas.NoTrans
            ldq = c_k
        one = self.__to_floats(1.0)
        zero = self.__to_floats(0.0)
        self.__cublas.gemm(self.__cublas.handle, Cublas.NoTrans, Trans, \
            c_n, c_m, c_k, one, dptr_v, c_n, dptr_q, ldq, zero, dptr_u, c_n)
        _try_calling(cuda.free(dptr_q))

    def add(self, other, s, a=None):
        n = self.dimension()
        m = other.nvec()
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        vsize = self.__dsize * n
        dptr_u = other.data_ptr()
        dptr_v = self.data_ptr()
        if numpy.isscalar(s):
            cublas_s = self.__to_floats(s)
            if a is None:
                nm = ctypes.c_int(n*m)
                inc = ctypes.c_int(1)
                self.__cublas.axpy \
                    (self.__cublas.handle, nm, cublas_s, dptr_u, inc, dptr_v, \
                     inc)
            else:
                k = a.shape[1]
                c_k = ctypes.c_int(k)
                if a.flags['C_CONTIGUOUS']:
                    Trans = Cublas.Trans
                    q = a
                    ldq = c_k
                elif a.flags['F_CONTIGUOUS']:
                    Trans = Cublas.NoTrans
                    q = a
                    ldq = c_m
                else:
                    q = numpy.ndarray(a.shape, dtype=a.dtype)
                    q[:,:] = a.copy()
                    Trans = Cublas.Trans
                    ldq = c_k
                hptr_q = ctypes.c_void_p(q.ctypes.data)
                dptr_q = ctypes.POINTER(ctypes.c_ubyte)()
                size_q = k*m*self.__dsize
                _try_calling(cuda.malloc(ctypes.byref(dptr_q), size_q))
                _try_calling(cuda.memcpy(dptr_q, hptr_q, size_q, cuda.memcpyH2D))
                one = self.__to_floats(1.0)
                self.__cublas.gemm(self.__cublas.handle, Cublas.NoTrans, Trans, \
                    c_n, c_k, c_m, cublas_s, dptr_u, c_n, dptr_q, ldq, \
                    one, dptr_v, c_n)
                _try_calling(cuda.free(dptr_q))
        else:
            for i in range(m):
                dptr_u = _shifted_ptr(other.data_ptr(), i*vsize)
                dptr_v = _shifted_ptr(self.data_ptr(), i*vsize)
                inc = ctypes.c_int(1)
                cublas_s = self.__to_floats(s[i])
                self.__cublas.axpy(self.__cublas.handle, c_n, cublas_s, \
                                   dptr_u, inc, dptr_v, inc)

    '''========== Other methods ====================================
    '''

    def __init__(self, arg, nvec=0, data_type=None, shallow=False): #, cublas=None):
        if isinstance(arg, Vectors):
            n = arg.dimension()
            m = arg.nvec()
            self.__is_complex = arg.__is_complex
            dtype = arg.data_type()
            dsize = arg.data_size()
            if shallow:
                self.__vdata = arg.vectors_data()
            else:
                size = n*m*dsize
                self.__vdata = _Data(size)
                _try_calling(cuda.memcpy(self.all_data_ptr(), arg.data_ptr(), \
                                         size, cuda.memcpyD2D))
        elif isinstance(arg, Matrix):
            if arg.order() != 'C_CONTIGUOUS':
                raise ValueError('Vectors data must be C_CONTIGUOUS')
            m, n = arg.shape()
            dtype = arg.data_type()
            dsize = arg.data_size()
            self.__is_complex = arg.is_complex()
            self.__vdata = arg.matrix_data() #TODO: deep copy case
        elif isinstance(arg, numpy.ndarray):
            m, n = arg.shape
            dtype = arg.dtype.type
            dsize = arg.itemsize
            size = n*m*dsize
            self.__vdata = _Data(size)
            ptr = ctypes.c_void_p(arg.ctypes.data)
            _try_calling(cuda.memcpy(self.all_data_ptr(), ptr, size, \
                                     cuda.memcpyH2D))
            self.__is_complex = \
                (dtype == numpy.complex64 or dtype == numpy.complex128)
        elif isinstance(arg, numbers.Number):
            dtype = data_type
            if dtype is None:
                dtype = numpy.float64
            if dtype == numpy.float32:
                dsize = 4
                self.__is_complex = False
            elif dtype == numpy.float64:
                dsize = 8
                self.__is_complex = False
            elif dtype == numpy.complex64:
                dsize = 8
                self.__is_complex = True
            elif dtype == numpy.complex128:
                dsize = 16
                self.__is_complex = True
            else:
                raise ValueError('data type %s not supported' % repr(dtype))
            n = arg
            m = nvec
            assert m >= 0
            if nvec > 0:
                size = int(n*m*dsize)
                self.__vdata = _Data(size)
                _try_calling(cuda.memset(self.all_data_ptr(), 0, size))
            else:
                self.__vdata = None
        else:
            raise ValueError \
                ('wrong argument %s in constructor' % repr(type(arg)))
        if cublas.handle is None:
            cublas.handle = Cublas(dtype)
        self.__selected = (0, m)
        self.__vdim = n
        self.__nvec = m
        self.__mvec = m
        self.__min_inc = 16
        self.__max_inc = 1024
        self.__dsize = dsize
        self.__dtype = dtype
        self.__cublas = cublas.handle

    def __float(self):
        dt = self.data_type()
        if dt == numpy.float32:
            return ctypes.c_float()
        elif dt == numpy.float64:
            return ctypes.c_double()
        else:
            raise ValueError('wrong data type %s passed to __float' % repr(dt))

    def __floats(self):
        dt = self.data_type()
        if dt == numpy.float32:
            floats = ctypes.c_float * 1
        elif dt == numpy.float64:
            floats = ctypes.c_double * 1
        elif dt == numpy.complex64:
            floats = ctypes.c_float * 2
        elif dt == numpy.complex128:
            floats = ctypes.c_double * 2
        else:
            raise ValueError('wrong data type %s passed to __complex' % repr(dt))
        return floats()

    def __to_floats(self, v):
        s = self.__floats()
        dt = self.data_type()
        if dt == numpy.float32:
            s[0] = ctypes.c_float(v)
        elif dt == numpy.float64:
            s[0] = ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            s[0] = v.real
            s[1] = v.imag
        else:
            raise ValueError('data type %s not supported' % repr(dt))
        return s

    def shape(self):
        return (self.__nvec, self.__vdim)

    def all_data_ptr(self):
        return self.__vdata.data_ptr()

    def data_ptr(self):
        n = self.dimension()
        vsize = n * self.__dsize
        return _shifted_ptr(self.all_data_ptr(), self.first() * vsize)
        
    def vectors_data(self):
        return self.__vdata

    def data_size(self):
        return self.__dsize

    def cublas(self):
        return self.__cublas

    def cublas_handle(self):
        return self.__cublas.handle

    def first(self):
        return self.__selected[0]

    def select_all(self):
        self.select(self.__nvec)

    def reference(self):
        v = Vectors(self, shallow=True)
        return v

    def is_complex(self):
        return self.__is_complex

    def conjugate(self):
        if not self.is_complex():
            return
        s = self.__to_floats(-1.0)
        m = self.nvec()
        n = self.dimension()
        inc = ctypes.c_int(2)
        ptr = _shifted_ptr(self.data_ptr(), self.data_size()//2)
        self.cublas().fscal(self.cublas().handle, m*n, s, ptr, inc)

    def orthogonalize(self, other):
        m = self.nvec()
        n = self.dimension()
        k = other.nvec()
        q = self.new_vectors(k, m)
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        c_k = ctypes.c_int(k)
        dptr_u = other.data_ptr()
        dptr_v = self.data_ptr()
        dptr_q = q.data_ptr()
        if self.is_complex():
            Trans = Cublas.ConjTrans
        else:
            Trans = Cublas.Trans
        zero = self.__to_floats(0.0)
        pl_one = self.__to_floats(1.0)
        mn_one = self.__to_floats(-1.0)
        self.__cublas.gemm(self.__cublas.handle, Trans, Cublas.NoTrans, \
            c_m, c_k, c_n, pl_one, dptr_v, c_n, dptr_u, c_n, zero, dptr_q, c_m)
        self.__cublas.gemm(self.__cublas.handle, Cublas.NoTrans, Trans, \
            c_n, c_m, c_k, mn_one, dptr_u, c_n, dptr_q, c_m, pl_one, dptr_v, c_n)
        return q

    def svd(self):
        q, sigma, w = scipy.linalg.svd(self.data(), full_matrices=False)
        self.fill(w)
        return sigma, _conjugate(q)
        m = self.nvec()
        n = self.dimension()
        c_n = ctypes.c_int(n)
        c_m = ctypes.c_int(m)
        c_lw = ctypes.c_int()
        err = self.__cublas.svd_buff_size(self.__cublas.cusolver_handle, \
            c_n, c_m, ctypes.byref(c_lw))
        lw = c_lw.value
        print('buffer size: %d' % lw)
        dptr_r = Cublas.CtypesPtr()
        dptr_s = Cublas.CtypesPtr()
        dptr_u = self.data_ptr()
        dptr_v = Cublas.CtypesPtr()
        dptr_w = Cublas.CtypesPtr()
        dptr_i = Cublas.CtypesPtr()
        dtype = self.data_type()
        dsize = self.data_size()
        _try_calling(cuda.malloc(ctypes.byref(dptr_w), lw*dsize))
        v = numpy.ndarray((m, m), dtype=dtype)
        _try_calling(cuda.malloc(ctypes.byref(dptr_v), m*m*dsize))
        if dtype == numpy.complex64:
            dtype = numpy.float32
            dsize = 4
        elif dtype == numpy.complex128:
            dtype = numpy.float64            
            dsize = 8
        sigma = numpy.ndarray((m,), dtype=dtype)
        _try_calling(cuda.malloc(ctypes.byref(dptr_r), m*dsize))
        _try_calling(cuda.malloc(ctypes.byref(dptr_s), m*dsize))
        _try_calling(cuda.malloc(ctypes.byref(dptr_i), 4))
        jobu = self.__cublas.ovwrt
        jobv = self.__cublas.all
        err = self.__cublas.svd(self.__cublas.cusolver_handle, jobu, jobv, \
            n, m, dptr_u, n, dptr_s, dptr_u, n, dptr_v, m, \
            dptr_w, lw, dptr_r, dptr_i)
#            c_n, c_m, dptr_u, c_n, dptr_s, dptr_u, c_n, dptr_v, c_m, \
#            dptr_w, c_lw, dptr_r, dptr_i)
        hptr_s = ctypes.c_void_p(sigma.ctypes.data)
        _try_calling(cuda.memcpy(hptr_s, dptr_s, m*dsize, cuda.memcpyD2H))
        hptr_v = ctypes.c_void_p(v.ctypes.data)
        _try_calling(cuda.memcpy(hptr_v, dptr_v, m*m*dsize, cuda.memcpyD2H))
        i = ctypes.c_int()
        _try_calling(cuda.memcpy(ctypes.byref(i), dptr_i, 4, cuda.memcpyD2H))
        _try_calling(cuda.free(dptr_r))
        _try_calling(cuda.free(dptr_s))
        _try_calling(cuda.free(dptr_v))
        _try_calling(cuda.free(dptr_w))
        _try_calling(cuda.free(dptr_i))
        return sigma, _conjugate(v)
#        return sigma, v.T

    def zero(self):
        m = self.nvec()
        if m < 1:
            return
        vsize = self.__dsize * self.__vdim
        _try_calling(cuda.memset(self.data_ptr(), 0, m*vsize))

    def fill(self, data):
        m, n = data.shape
#        if m != self.__nvec or n != self.__vdim:
        if m != self.nvec() or n != self.__vdim:
            raise ValueError('mismatching dimensions in fill()')
        if m < 1:
            return
        dtype = data.dtype.type
        if dtype != self.__dtype:
            raise ValueError('mismatching data types in fill()')
        if data.flags['C_CONTIGUOUS']:
            the_data = data
        else:
            the_data = numpy.ndarray((m, n), dtype=dtype)
            the_data[:,:] = data
        size = m * self.__dsize * self.__vdim
        ptr = ctypes.c_void_p(the_data.ctypes.data)
        _try_calling(cuda.memcpy(self.data_ptr(), ptr, size, cuda.memcpyH2D))
#        _try_calling(cuda.memcpy(self.all_data_ptr(), ptr, size, cuda.memcpyH2D))

    def data(self):
        m = self.nvec()
        n = self.dimension()
        v = numpy.ndarray((m, n), dtype=self.data_type())
        if m < 1:
            return v
        hptr_v = ctypes.c_void_p(v.ctypes.data)
        size = int(n * m * self.__dsize)
        _try_calling(cuda.memcpy(hptr_v, self.data_ptr(), size, cuda.memcpyD2H))
        return v

    def asarray(self):
        return self.data().T


class Matrix:

    def __init__(self, arg):
        if isinstance(arg, Vectors):
            n = arg.dimension()
            m = arg.nvec()
            self.__shape = (m, n)
            self.__dtype = arg.data_type()
            self.__dsize = arg.data_size()
            self.__is_complex = arg.is_complex()
            self.__order = 'C_CONTIGUOUS'
            self.__mdata = arg.vectors_data()
        elif isinstance(arg, numpy.ndarray):
            self.__shape = arg.shape
            self.__dtype = arg.dtype.type
            self.__dsize = arg.itemsize
            self.__is_complex = (arg.dtype.kind == 'c')
            if arg.flags['C_CONTIGUOUS']:
                self.__order = 'C_CONTIGUOUS'
            elif arg.flags['F_CONTIGUOUS']:
                self.__order = 'F_CONTIGUOUS'
            else:
                msg = 'Matrix data must be either C- or F-contiguous'
                raise ValueError(msg)
            m, n = self.__shape
            size = n*m*self.__dsize
            self.__mdata = _Data(size)
            ptr = ctypes.c_void_p(arg.ctypes.data)
            _try_calling(cuda.memcpy(self.data_ptr(), ptr, size, cuda.memcpyH2D))
        else:
            raise ValueError \
                ('wrong argument %s in Matrix constructor' % repr(type(arg)))

    def __floats(self):
        dt = self.__dtype
        if dt == numpy.float32:
            floats = ctypes.c_float * 1
        elif dt == numpy.float64:
            floats = ctypes.c_double * 1
        elif dt == numpy.complex64:
            floats = ctypes.c_float * 2
        elif dt == numpy.complex128:
            floats = ctypes.c_double * 2
        else:
            raise ValueError('wrong data type %s passed to __complex' % repr(dt))
        return floats()

    def __to_floats(self, v):
        s = self.__floats()
        dt = self.__dtype
        if dt == numpy.float32:
            s[0] = ctypes.c_float(v)
        elif dt == numpy.float64:
            s[0] = ctypes.c_double(v)
        elif dt == numpy.complex64 or dt == numpy.complex128:
            s[0] = v.real
            s[1] = v.imag
        else:
            raise ValueError('data type %s not supported' % repr(dt))
        return s
        
    def data_ptr(self):
        return self.__mdata.data_ptr()

    def matrix_data(self):
        return self.__mdata

    def order(self):
        return self.__order

    def shape(self):
        return self.__shape

    def data_type(self):
        return self.__dtype

    def data_size(self):
        return self.__dsize

    def is_complex(self):
        return self.__is_complex

    def fill(self, data):
        m, n = self.__shape
        size = n*m*self.__dsize
        ptr = ctypes.c_void_p(data.ctypes.data)
        _try_calling(cuda.memcpy(self.data_ptr(), ptr, size, cuda.memcpyH2D))

    def dots(self):
        v = Vectors(self, shallow=True)
        return v.dots(v)

    def new_vectors(self, dim=None, nv=0):
        if dim is None:
            dim = self.shape()[1]
        return Vectors(dim, nv, self.data_type())

    def apply(self, x, y, transp=False):
        if x.data_type() != self.__dtype or y.data_type() != self.__dtype:
            raise ValueError('Matrix and vectors data types differ')
        m, n = self.__shape
        if transp:
            if n != y.dimension() or m != x.dimension():
                raise ValueError('Matrix and vectors dimensions incompatible')
        else:
            if m != y.dimension() or n != x.dimension():
                raise ValueError('Matrix and vectors dimensions incompatible')
        k = x.nvec()
        if k != y.nvec():
            raise ValueError('Numbers of input and output vectors differ')
        k = ctypes.c_int(k)
        nx = ctypes.c_int(x.dimension())
        ny = ctypes.c_int(y.dimension())
        conj = False
        if self.__order == 'C_CONTIGUOUS':
            if transp:
                if self.__is_complex:
                    x.conjugate()
                    conj = True
                Trans = Cublas.NoTrans
            else:
                Trans = Cublas.Trans
            lda = ctypes.c_int(n)
        elif self.__order == 'F_CONTIGUOUS':
            if transp:
                if self.__is_complex:
                    Trans = Cublas.ConjTrans
                else:
                    Trans = Cublas.Trans
            else:
                Trans = Cublas.NoTrans
            lda = ctypes.c_int(m)
        one = self.__to_floats(1.0)
        zero = self.__to_floats(0.0)
        dptr_a = self.data_ptr()
        dptr_x = x.data_ptr()
        dptr_y = y.data_ptr()
        x.cublas().gemm(x.cublas_handle(), Trans, Cublas.NoTrans, \
            ny, k, nx, one, dptr_a, lda, dptr_x, nx, zero, dptr_y, ny)
        if conj:
            x.conjugate()
            y.conjugate()


def _try_calling(err):
    if err != 0:
        raise RuntimeError('cuda error %d' % err)


def _shifted_ptr(dev_ptr, shift=0):
    ptr = ctypes.cast(dev_ptr, ctypes.c_void_p)
    return ctypes.cast(ptr.value + shift, ctypes.POINTER(ctypes.c_ubyte))


def _ptr_value(dev_ptr):
    ptr = ctypes.cast(dev_ptr, ctypes.c_void_p)
    return ptr.value


def _conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj()
    else:
        return a


class _Data:

    def __init__(self, size):
        self.__data = ctypes.POINTER(ctypes.c_ubyte)()
        _try_calling(cuda.malloc(ctypes.byref(self.__data), size))

    def __del__(self):
        _try_calling(cuda.free(self.__data))

    def data_ptr(self):
        return self.__data
