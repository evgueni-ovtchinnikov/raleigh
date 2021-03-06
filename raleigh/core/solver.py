# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''
RALEIGH (RAL EIGensolvers for real symmetric and Hermitian problems) core
solver.

For advanced users only - consider using more user-friendly interfaces in 
raleigh/interfaces first.

Implements a block Conjugate Gradient algorithm for the computation of 
several eigenpairs (eigenvalues and corresponding eigenvectors) of real 
symmetrtic or Hermitian problems, namely:
- standard eigenvalue problem 
  A x = lambda x
- generalized eigenvalue problems
  A x = lambda B x
  A B x = lambda x
where A and B are real symmetric or Hermitian operators, B being positive
definite.

The algorithm operates on sets of vectors (v_1, ..., v_m) encapsulated by
an abstract data type Vectors with the following methods:
---------------------------------
new_vectors(self, nv=0, dim=None)
  returns a new Vectors object encapsulating nv vectors of dimension dim
  if dim is not None or else the same dimension as self
---------------------------------
dimension(self):
  returns the dimension of vectors encapsulated by self
---------------------------------
select(self, nv, first=0)
  selects a subset of nv encapsulated vectors starting from first;
  all subsequent operations on self will involve these vectors only
---------------------------------
selected(self)
  returns current subset selection as tuple (first, nv)
---------------------------------
clone(self)
  returns a copy of (selected part of) self
---------------------------------
append(self, other):
  appends vectors from Vectors object other to those of self
---------------------------------
nvec(self)
  returns the number of currently selected vectors
---------------------------------
data_type(self)
  returns the data type of vectors' elements 
  (numpy.float32, numpy.float64, numpy.complex64 or numpy.complex128)
---------------------------------
fill_random(self):
  fills vectors with random values uniformly distributed between -1 and 1
---------------------------------
copy(self, other, ind=None)
  copies vectors from self to other:
  if ind is None, previously selected vectors of self copied into previously
  selected vectors of other (the numbers of selected vectors must coincide),
  otherwise vectors specified by the array of indices ind are copied
---------------------------------
scale(self, s, multiply=False)
  if multiply is True, previously selected vectors of self are multiplied by
  the respective elements of numpy ndarray s, otherwise the former are divided
  by the latter, skipping division for zero elements
---------------------------------
dots(self, other, transp=False)
  if transp is False, returns the numpy ndarray of dot products of previously
  selected vectors of self with the respective selected vectors of other,
  otherwise returns a numpy ndarray, i-th element of which is the dot product
  of the vector of i-th components of selected vectors of self by the vector
  of i-th components of selected vectors of other (note that for complex 
  vectors the complex dot products are computed, i.e. complex conjugation is
  applied to the components of vectors in other)
---------------------------------
dot(self, other)
  returns the ndarray of shape (m, n), where m and n are the numbers of the
  selected vectors in other and self respectively, containing dot products of
  the selected vectors of self by those of other (again, dot products are
  complex in the complex vectors case)
---------------------------------
multiply(self, q, other)
  for each column q[:, j] of ndarray q, assigns the linear combination of the
  selected vectors of self with coefficients q[0, j], q[1, j], ... to j-th
  selected vector of other
---------------------------------
add(self, other, s, q=None)
  if s is a scalar: 
    if q is None, adds the selected vectors of other multiplied by s to the 
    respective selected vectors of self, otherwise
    for each column q[:, j] of ndarray q, adds the linear combination of 
    selected vectors of other with coefficients q[0, j], q[1, j], ... 
    multiplied by s to j-th selected vector of self,
  otherwise adds the selected vectors of other multiplied by the respective
  elements of one-dimensional ndarray s to respective selected vectors of self
  (q is ignored)
---------------------------------

The folder raleigh/algebra contains three implementations of Vectors type:
(i) numpy implementaion, (ii) mkl implementation (requires MKL 10.3 or later:
needs mkl_rt.dll on Windows, libmkl_rt.so on Linux), and (iii) CUDA GPU 
implementation (requires CUDA-enabled GPU and NVIDIA Toolkit). These may be
used as templates for further implementations - MPI, out of core etc.

The number of wanted eigenpairs does not need to be set before calling the 
core solver - instead, the user may provide an object responsible for stopping 
the computation based on the data computed so far. 

An object responsible for deciding whether a particular eigenpair has converged 
can also be supplied by the user (default convergence criteria object is 
available).

If some eigenvectors are already available, they can be passed to the core 
solver, which then will compute further eigenpairs. Initial guesses to
eigenvectors may also be supplied by the user.
'''

import math
import numpy
import numpy.linalg as nla
import scipy.linalg as sla

RECORDS = 100


class DefaultConvergenceCriteria:
    '''
    Convergence criteria to be used if not specified by the user via Options
    (see Options.convergence_criteria below).
    '''
    def __init__(self):
        self.tolerance = 1e-3
        self.error = 'kinematic eigenvector error'
    def set_error_tolerance(self, error, tolerance):
        self.error = error
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data(self.error, i)
        return err >= 0 and err <= self.tolerance


class Options:
    '''
    Solver options.
    
    Attributes
    ----------
    verbosity : int
        printout level
       <0 : no output
        0 : error an warning messages
        1 : iteration number, converged eigenvalues
        2 : convergence data for all current iterates
    max_iter : int
        maximal number of iterations per eigenpair;
        if negative, set by solver
    min_iter : int
        minimal number of iterations per eigenpair
    block_size : int
        the nuber of simultaneously iterated eigenvector approximations;
        if negative, set by solver
    threads : int
        the number of CPU threads, to be used to determine the block size if
        not set by the user
    sigma : float
        if not None, indicates that the solver is used in shift-invert context
    convergence_criteria : object
        if not None, must be an object with method satisfied(self, solver, i)
        that returns True if i-th approximate eigenpair converged and False
        otherwise, based on the information provided by solver.convergence_data
        (see below)
    stopping_criteria : object
        if not None, must be an object with method satisfied(self, solver)
        that returns True if sufficient number of eigenpairs have been computed
        and False otherwise, based on the values of solver attributes (e.g.
        solver.eigenvalues) and possibly the user input, see 
        interfaces.partial_svd.DefaultStoppingCriteria for an example
    detect_stagnation : bool
        if set to True, detects the loss of convergence, i.e. impossibility to
        significantly improve the accuracy of the approximation (set to False
        and request very high accuracy if you want to test the solver for 
        numerical stability)
    max_quota : float
        the iterations will stop as soon as the number of computed eigenpairs 
        exceeds max_quota multiplied by the problem size, and the rest of 
        eigenpairs will be computed by scipy.linalg.eigh
    '''
    def __init__(self):
        self.verbosity = 0
        self.max_iter = -1
        self.min_iter = 0
        self.block_size = -1
        self.threads = -1
        self.sigma = None
        self.convergence_criteria = None
        self.stopping_criteria = None
        self.detect_stagnation = True
        self.max_quota = 0.75


class EstimatedErrors:
    '''
    Estimated errors container.
    
    Attributes
    ----------
    kinematic : one-dimensional ndarray of floats
        error estimates based on the convergence history
    residual : one-dimensional ndarray of floats
        error estimates based on the residuals
    '''
    def __init__(self):
        self.kinematic = numpy.ndarray((0,), dtype=numpy.float32)
        self.residual = numpy.ndarray((0,), dtype=numpy.float32)
    def __getitem__(self, item):
        return self.kinematic[item], self.residual[item]
    def append(self, est):
        self.kinematic = numpy.concatenate((self.kinematic, est[0, :]))
        self.residual = numpy.concatenate((self.residual, est[1, :]))
    def reorder(self, ind):
        self.kinematic = self.kinematic[ind]
        self.residual = self.residual[ind]


class Problem:
    '''
    Eigenvalue problem specification.
    
    Attributes
    ----------
    __A : object
        operator A (stiffness matrix)
    __B : object
        operator B (mass matrix)
    __type : string
        problem type
        'std' :  standard A x = lambda x
        'gen' :  generalized A x = lambda B x
        'pro' :  generalized A B x = lambda x
    '''
    def __init__(self, v, A, B=None, prod=None):
        self.__vector = v
        self.__A = A
        self.__B = B
        if B is None:
            self.__type = 'std'
        else:
            if prod is None:
                self.__type = 'gen'
            else:
                self.__type = 'pro'
    def A(self):
        return self.__A
    def B(self):
        return self.__B
    def type(self):
        return self.__type[0]
    def vector(self):
        return self.__vector


class Solver:
    '''
    Eigenvalue problem solver specification.
    
    Attributes
    ----------
    __problem : instance of Problem
        problem specification
    __P : object
        preconditioner
    iteration : int
        the current iteration number
    lcon : int
        the number of computed eigenpairs on the left margin of the spectrum
    rcon : int
        the number of computed eigenpairs on the right margin of the spectrum
    eigenvalues : one-dimensional ndarray of dtype numpy.float64
        converged eigenvalues
    eigenvalue_errors : one-dimensional ndarray of dtype EstimatedErrors
        estimated errors for computed eigenvalues
    eigenvector_errors : one-dimensional ndarray of dtype EstimatedErrors
        estimated errors for computed eigenvectors
    residual_norms : one-dimensional ndarray of dtype numpy.float32
        residual norms for computed eigenpairs
    convergence_status : one-dimensional ndarray of dtype numpy.int32
        computed eigenpairs convergence status
        > 0 : the number of iterations taken to converge
        < 0 : the negative of the number of iterations taken to stagnate
    cnv : one-dimensional ndarray of dtype numpy.int32
        current convergence status
            0 : has not converged yet
        i > 0 : converged after i iterations
        i < 0 : stopped converging after i iterations
    lmd : one-dimensional ndarray of dtype numpy.float64
        current eigenvalue iterates
    res : one-dimensional ndarray of dtype numpy.float32
        current residuals
    err_lmd : two-dimensional ndarray of dtype numpy.float32
        error estimates for current eigenvalue iterates
    err_X : two-dimensional ndarray of dtype numpy.float32
        error estimates for current eigenvector iterates
    '''
    def __init__(self, problem):
        self.__problem = problem
        self.__P = None
        self.iteration = 0
        self.lcon = 0
        self.rcon = 0
        self.eigenvalues = numpy.ndarray((0,), dtype=numpy.float64)
        self.eigenvalue_errors = EstimatedErrors()
        self.eigenvector_errors = EstimatedErrors()
        self.residual_norms = numpy.ndarray((0,), dtype=numpy.float32)
        self.convergence_status = numpy.ndarray((0,), dtype=numpy.int32)
        # data to be set by solver
        self.eigenvectors = None
        self.eigenvectors_im = None
        self.block_size = None
        self.cnv = None
        self.lmd = None
        self.res = None
        self.err_lmd = None
        self.err_X = None

    def set_preconditioner(self, P):
        self.__P = P

    def problem(self):
        return self.__problem

    def preconditioner(self):
        return self.__P

    def convergence_data(self, what='residual', which=0):
        '''Reports current convergence data.
        
        Parameters
        ----------
        what : string
            convergence data to report (can be abbreviated, full names below)
        which : int
            for which eigenpair iterate to report
        '''
        if what.find('block') > -1:
            '''block size
            '''
            return self.block_size
        elif what.find('res') > -1 and what.find('vec') == -1:
            '''relative residual.
            
            WARNING: use only for the largest eigenvalues computation.
            '''
            max_lmd = numpy.amax(abs(self.lmd))
            if self.lcon + self.rcon > 0:
                max_lmd = max(max_lmd, numpy.amax(abs(self.eigenvalues)))
            return self.res[which]/max_lmd
        elif what.find('val') > -1:
            if what.find('max') > -1:
                max_lmd = numpy.amax(abs(self.lmd))
                if self.lcon + self.rcon > 0:
                    max_lmd = max(max_lmd, numpy.amax(abs(self.eigenvalues)))
                return max_lmd
            if what.find('err') > -1:
                err = self.err_lmd[:, which]
                if what.find('k'):
                    '''kinematic eigenvalue error estimate.
                    '''
                    return err[0]
                else:
                    '''residual-based eigenvalue error estimate.
                    '''
                    return err[1]
            else:
                '''current eigenvalue iterate.
                '''
                return self.lmd[which]
        elif what.find('vec') > -1:
            err = self.err_X[:, which]
            if what.find('k') > -1:
                '''kinematic eigenvector error estimate.
                '''
                return err[0]
            else:
                '''residual-based eigenvector error estimate.
                '''
                return err[1]
        else:
            raise ValueError('convergence data %s not found' % what)

    def solve(self, eigenvectors, options=Options(), which=(-1, -1), \
              extra=(-1, -1), init=(None, None)):
        '''Main core solver routine.
        
        Parameters
        ----------
        eigenvectors : object of abstract Vectors type
            eigenvectors container; normally empty (eigenvectors.nvec() = 0),
            if not, then assumed by the solver to contain previously computed
            eigenvectors of the solved problem; on return contains all computed
            eigenvectors (previous and new)
        options : object of type Options
            solver options
        which : int or tuple of two ints
            if int, the number of the largest eigenvalues wanted;
            if tuple, the numbers of eigenvalues wanted on the left (which[0])
            and right (which[1]) margine of the spectrum;
            negative values mark unknown number of wanted eigenvalues: in this
            case, the user must provide stopping_criteria
        extra : tuple of two ints
            numbers of extra eigenvectors corresponding to eigenvalues on the
            margins of the spectrum to iterate purely for the sake of better
            convergence: convergence criteria will not be applied to these
            extra eigenpairs, i.e. iterations will stop when all wanted
            eigenpairs converge
        init : tuple of two Vectors objects
            each tuple item, if not None, contains initial guesses for
            eigenvectors corresponding to eigenvalues on the respective margin
            of the spectrum
            
        Returns
        -------
        status : int
            execution status
            0 : success
            1 : maximal number of iterations exceeded
            2 : no search directions left (bad problem data or preconditioner)
            3 : some of the requested left eigenvalues may not exist
            4 : some of the requested right eigenvalues may not exist
           <0 : fatal error - exception thrown
        '''

        verb = options.verbosity

        try:
            l = len(which)
            if l != 2:
                raise ValueError\
                      ('which must be either integer or tuple of 2 integers')
            largest = False
        except:
            largest = True

        if largest:
            if which >= 0:
                left = which//2
                right = which - left
            else:
                left = -1
                right = -1
        else:
            left = int(which[0])
            right = int(which[1])
        if left == 0 and right == 0:
            if verb > -1:
                print('No eigenpairs requested, quit')
            return 0

        m = int(options.block_size)
        if m < 0:
            m = _default_block_size(left, right, extra, init, options.threads)
        else:
            if (left == 0 or right == 0) and not largest:
                if m < 3:
                    if verb > -1:
                        print('Block size %d too small, will use 3 instead' % m)
                    m = 3
            else:
                if m < 4:
                    if verb > -1:
                        print('Block size %d too small, will use 4 instead' % m)
                    m = 4
        self.block_size = m

        n = eigenvectors.dimension()

        #output
        self.iteration = 0
        self.lcon = 0
        self.rcon = 0
        self.eigenvalues = numpy.ndarray((0,), dtype=numpy.float64)
        self.eigenvalue_errors = EstimatedErrors()
        self.eigenvector_errors = EstimatedErrors()
        self.residual_norms = numpy.ndarray((0,), dtype=numpy.float32)
        self.convergence_status = numpy.ndarray((0,), dtype=numpy.int32)

        if m < n//2:
            try:
                status = self._solve(eigenvectors, options, which, extra, init)
                if status > 1:
                    if verb > -1:
                        print('core solver return status %d' % status)
                    return status - 1
            except _Error as err:
                if verb > -1:
                    print('%s' % err.value)
                return -1
        else:
            status = 1

        if status == 0:
            return 0 # success

        Xc = eigenvectors
        nc = Xc.nvec()
        m = n - nc
        if verb > -1:
            msg = '%d eigenpairs not computed by CG, applying ' + \
                  'Rayleigh-Ritz procedure'
            print( msg % m)
            print('in the complement subspace...')

        X = eigenvectors.new_vectors(m)
        X.fill_random()
        Y = X.new_vectors(m)
        Z = X.new_vectors(m)
        std = (self.__problem.type() == 's')
        pro = (self.__problem.type() == 'p')

        opA = self.problem().A()
        A = lambda x, y: opA.apply(x, y)
        opB = self.problem().B()
        if opB is not None:
            B = lambda x, y: opB.apply(x, y)
        data_type = eigenvectors.data_type()

        if nc > 0:
            if not std:
                BXc = eigenvectors.clone()
                if nc > 0:
                    B(Xc, BXc)
            else:
                BXc = Xc
            if nc > 0:
                Gc = BXc.dot(Xc)
                #Gci = nla.inv(Gc)
                # approximate inverse of Gc
                Gci = 2*numpy.identity(nc, dtype=data_type) - Gc
            Q = numpy.dot(Gci, X.dot(BXc))
            X.add(Xc, -1.0, Q)
            Q = numpy.dot(Gci, X.dot(BXc))
            X.add(Xc, -1.0, Q)

        if not std:
            B(X, Y)
            XBX = Y.dot(X)
        else:
            XBX = X.dot(X)
        lmd, Q = sla.eigh(-XBX)
        lmd = -lmd
        epsilon = 100*numpy.finfo(data_type).eps
        k = numpy.sum(lmd <= epsilon*lmd[0])
        if k > 0:
            if verb > -1:
                #print(lmd[-k], lmd[0])
                msg = 'dropping %d linear dependent vectors ' + \
                      'from the Rayleigh-Ritz procedure...'
                print(msg % k)
            X.multiply(Q, Z)
            Z.copy(X)
            Y.multiply(Q, Z)
            Z.copy(Y)
            m -= k
            X.select(m)
            Y.select(m)
            Z.select(m)
            if not std:
                B(X, Y)
                XBX = Y.dot(X)
            else:
                XBX = X.dot(X)

        if pro:
            A(Y, Z)
            XAX = Z.dot(Y)
        else:
            A(X, Z)
            XAX = Z.dot(X)

        lmdx, Q = sla.eigh(XAX, XBX, turbo=False, overwrite_a=True, \
                           overwrite_b=True)
        X.multiply(Q, Z)
        Z.copy(X)
        eigenvectors.append(X)
        self.eigenvalues = numpy.concatenate((self.eigenvalues, lmdx))

        return 0

    def _solve(self, eigenvectors, options, which, extra, init):

        verb = options.verbosity
        sigma = options.sigma

        try:
            l = len(which)
            if l != 2:
                raise ValueError\
                      ('which must be either integer or tuple of 2 integers')
            largest = False
            left = int(which[0])
            right = int(which[1])
        except:
            largest = True
            left = which
            right = which

        m = self.block_size
        if left == 0 and not largest:
            r = 0.0
            l = 1
        elif right == 0:
            r = 1.0
            l = m - 1
        elif left > 0 and right > 0:
            r = left/(left + 1.0*right)
            l = int(round(r*m))
            if l < 2:
                l = 2
            if l > m - 2:
                l = m - 2
        else:
            r = 0.5
            l = m//2
        left_ratio = r
        block_size = m
        left_block_size = l

        extra_left = int(extra[0])
        extra_right = int(extra[1])
        if left >= 0:
            if extra_left > 0:
                left_total = left + extra_left
            else:
                left_total = max(left + 1, left_block_size)
            if verb > 2:
                print('left total: %d' % left_total)
        if right >= 0:
            if extra_right > 0:
                right_total = right + extra_right
            else:
                right_total = max(right + 1, block_size - left_block_size)
            if verb > 2:
                print('right_total: %d' % right_total)
        if verb > 0:
            print('left block size %d, right block size %d' % (l, m - l))

        # problem
        problem = self.__problem
        vector = problem.vector()
        problem_type = problem.type()
        std = (problem_type == 's')
        gen = (problem_type == 'g')
        pro = (problem_type == 'p')
        data_type = vector.data_type()
        epsilon = numpy.finfo(data_type).eps
        single = (data_type == numpy.float32 or data_type == numpy.complex64)

        # convergence data
        self.cnv = numpy.zeros((m,), dtype=numpy.int32)
        self.lmd = numpy.zeros((m,), dtype=numpy.float64)
        self.res = -numpy.ones((m,), dtype=numpy.float32)
        self.err_lmd = -numpy.ones((2, m,), dtype=numpy.float32)
        self.err_X = -numpy.ones((2, m,), dtype=numpy.float32)

        # convergence criteria
        if options.convergence_criteria is None:
            convergence_criteria = DefaultConvergenceCriteria()
        else:
            convergence_criteria = options.convergence_criteria

        # convergence history data
        iterations = numpy.zeros((m,), dtype=numpy.int32)
        dlmd = numpy.zeros((m, RECORDS), dtype=numpy.float32)
        dX = numpy.ones((m,), dtype=numpy.float32)
        acf = numpy.ones((2, m,), dtype=numpy.float32)
        cluster = numpy.zeros((2, m), dtype=numpy.int32)

        # workspace
        X = vector.new_vectors(m)
        X.fill_random()
        Y = vector.new_vectors(m)
        Z = vector.new_vectors(m)
        W = vector.new_vectors(m)
        AX = vector.new_vectors(m)
        AY = vector.new_vectors(m)
        if not std:
            BX = vector.new_vectors(m)
            BY = vector.new_vectors(m)
        else:
            BX = X
            BY = Y
        AZ = AY
        BZ = BY

        # copy initial vectors if present
        l = left_block_size
        m = block_size
        init_lX = init[0]
        if init_lX is not None:
            init_left = min(l, init_lX.nvec())
            X.select(init_left)
            init_lX.select(init_left)
            init_lX.copy(X)
        else:
            init_left = 0
        init_rX = init[1]
        if init_rX is not None:
            init_right = min(m - l, init_rX.nvec())
            X.select(init_right, init_left)
            init_rX.select(init_right)
            init_rX.copy(X)

        # check for zero initial vectors
        X.select(m)
        s = X.dots(X)
        for i in range(m):
            if s[i] == 0.0:
                if verb > -1:
                    print('Zero initial guess, replacing with random')
                X.select(1, i)
                X.fill_random()
                s[i : i + 1] = X.dots(X)
        X.select(m)
        s = numpy.sqrt(X.dots(X))
        X.scale(s)

        # shorcuts
        detect_stagn = options.detect_stagnation
        lmd = self.lmd
        res = self.res
        err_lmd = self.err_lmd
        err_X = self.err_X
        A = lambda x, y: problem.A().apply(x, y)
        opB = problem.B()
        if opB is not None:
            B = lambda x, y: opB.apply(x, y)
        else:
            B = None
        opP = self.__P
        if opP is not None:
            P = lambda x, y: opP.apply(x, y)
        else:
            P = None

        # constraints (already available eigenvectors e.g. from previous run)
        self.eigenvectors = eigenvectors
        Xc = eigenvectors
        nc = Xc.nvec()
        if not std:
            BXc = eigenvectors.clone()
            if nc > 0:
                B(Xc, BXc)
            self.eigenvectors_im = BXc
        else:
            BXc = Xc
        if nc > 0:
            Gc = BXc.dot(Xc)
            # approximate inverse of Gc
            Gci = 2*numpy.identity(nc, dtype=data_type) - Gc

        # initialize
        leftX = left_block_size
        rightX = block_size - leftX
        rec = 0
        ix = 0 # first X
        nx = block_size
        ny = block_size
        nz = 0
        lmdz = None

        if Xc.nvec() > 0:
            # orthogonalize X to Xc
            # std: X := X - Xc Xc* X
            # gen: X := X - Xc BXc* X
            # pro: X := X - Xc BXc* X
            Q = numpy.dot(Gci, X.dot(BXc))
            X.add(Xc, -1.0, Q)

        if not std:
            B(X, BX)
        XBX = BX.dot(X)

        # do pivoted Cholesky for XBX to eliminate linear dependent X
        U = XBX.copy()
        ind, dropped = _piv_chol(U, 0, 1e-2)
        if dropped > 0:
            if verb > 0:
                print('dropped %d initial vectors out of %d' % (dropped, nx))
            # drop linear dependent initial vectors
            nx -= dropped
            if nx > 0:
                W.select(nx)
                X.copy(W, ind)
                W.copy(X)
            X.select(dropped, nx)
            X.fill_random()
            if not std:
                if nx > 0:
                    BX.copy(W, ind)
                    W.copy(BX)
                BX.select(dropped, nx)
                B(X, BX)
            if Xc.nvec() > 0:
                # orthogonalize X to Xc
                Q = numpy.dot(Gci, X.dot(BXc))
                Xc.multiply(Q, W)
                X.add(W, -1.0)
                if not std:
                    BXc.multiply(Q, W)
                    BX.add(W, -1.0)
            nx = m
            X.select(nx)
            if not std:
                BX.select(nx)
            XBX = BX.dot(X)

        # Rayleigh-Ritz in the initial space
        if pro:
            A(BX, AX)
            XAX = AX.dot(BX)
        else:
            A(X, AX)
            XAX = AX.dot(X)
        lmdx, Q = sla.eigh(XAX, XBX, turbo=False)
        W.select(m)
        X.multiply(Q, W)
        W.copy(X)
        AX.multiply(Q, W)
        W.copy(AX)
        if not std:
            BX.multiply(Q, Z)
            Z.copy(BX)

# ===== main CG loop
        max_iter = options.max_iter
        min_iter = options.min_iter
        if max_iter < 0:
            max_iter = 100
        self.iteration = 0
        while True:

            maxit = 0
            if left != 0 and left_block_size > 0:
                maxit = numpy.amax(iterations[:left_block_size])
            if right != 0 and left_block_size < block_size:
                maxit = max(maxit, numpy.amax(iterations[left_block_size:]))
            if maxit >= max_iter:
                if verb > -1:
                    msg = 'iterations limit of %d exceeded, terminating'
                    print(msg % max_iter)
                break

            if verb > 0:
                print('------------- iteration %d' % self.iteration)

            if pro:
                XAX = AX.dot(BX)
            else:
                XAX = AX.dot(X)
            XBX = BX.dot(X)
            da = XAX.diagonal()
            db = XBX.diagonal()
            new_lmd = _real(da/db)

            # estimate error in residual computation due to the error in
            # computing AX, to be used in detecting convergence stagnation
            Lmd = numpy.zeros((nx, nx))
            Lmd[range(nx), range(nx)] = lmdx #new_lmd
            RX = XAX - numpy.dot(XBX, Lmd)
            delta_R = _norm(RX, 0)
            if gen:
                s = numpy.sqrt(abs(X.dots(X)))
                delta_R /= s
            rv_err = numpy.amax(abs(new_lmd - lmdx))/numpy.amax(abs(lmdx))
            rv_no = numpy.amax(abs(XBX - numpy.eye(nx)))
            if verb > 2:
                print('Ritz values error: %.1e' % rv_err)
                print('Ritz vectors non-orthonormality: %.1e' % rv_no)
            if max(rv_err, rv_no) > math.sqrt(epsilon):
                if verb > 0:
                    if verb < 3:
                        print('Ritz values error: %.1e' % rv_err)
                        print('Ritz vectors non-orthonormality: %.1e' % rv_no)
                    print('restarting...')
                rec = 0
                nz = 0
                sigma, Q = X.svd()
                if std:
                    XBX = X.dot(X)
                else:
                    B(X, BX)
                    XBX = BX.dot(X)
                if pro:
                    A(BX, AX)
                    XAX = AX.dot(BX)
                else:
                    A(X, AX)
                    XAX = AX.dot(X)
                #rv_no = numpy.amax(abs(XBX - numpy.eye(nx)))
                #print('Ritz vectors non-orthonormality: %.1e' % rv_no)
                lmdx, Q = sla.eigh(XAX, XBX, turbo=False)
                W.select(nx)
                X.multiply(Q, W)
                W.copy(X)
                AX.multiply(Q, W)
                W.copy(AX)
                if not std:
                    BX.multiply(Q, W)
                    W.copy(BX)
                if pro:
                    XAX = AX.dot(BX)
                else:
                    XAX = AX.dot(X)
                if std:
                    XBX = X.dot(X)
                else:
                    XBX = BX.dot(X)
                rv_no = numpy.amax(abs(XBX - numpy.eye(nx)))
                #print('Ritz vectors non-orthonormality: %.1e' % rv_no)
                da = XAX.diagonal()
                db = XBX.diagonal()
                new_lmd = _real(da/db)

            for i in range(nx):
                iterations[ix + i] += 1
#            if self.iteration > 0:
            if rec > 0:
                # compute eigenvalue decrements
                for i in range(nx):
                    if iterations[ix + i]:
                        delta = lmd[ix + i] - new_lmd[i]
                        eps = math.sqrt(epsilon)
                        eps *= max(abs(lmd[ix + i]), abs(new_lmd[i]))
                        if abs(delta) > eps:
                            dlmd[ix + i, rec - 1] = delta
#                    iterations[ix + i] += 1
                if verb > 3:
                    print('eigenvalues shifts history:')
                    print(numpy.array_str(dlmd[ix : ix + nx, :rec].T, \
                                          precision=2))

            lmd[ix : ix + nx] = new_lmd

            # compute residuals
            # std: A X - X lmd
            # gen: A X - B X lmd
            # pro: A B X - X lmd
            W.select(nx, ix)
            Y.select(nx)
            AX.copy(W)
            if gen:
                W.add(BX, -lmd[ix : ix + nx])
            else:
                W.add(X, -lmd[ix : ix + nx])

            if Xc.nvec() > 0:
                # orthogonalize W to Xc
                # std: W := W - Xc Xc* W
                # gen: W := W - BXc Xc* W
                # pro: W := W - Xc BXc* W
                if pro:
                    Q = numpy.dot(Gci, W.dot(BXc))
                else:
                    Q = numpy.dot(Gci, W.dot(Xc))
                if gen:
                    W.add(BXc, -1.0, Q)
                else:
                    W.add(Xc, -1.0, Q)

            if pro:
                W.copy(Y)
                B(Y, W)
                s = W.dots(Y)
            else:
                s = W.dots(W)
            res[ix : ix + nx] = numpy.sqrt(abs(s))

            # kinematic error estimates
            if rec > 3: # sufficient history available
                for i in range(nx):
                    if dX[ix + i] > 0.01:
                        err_X[0, ix + i] = -1.0
                        continue
                    k = 0
                    s = 0
                    # go through the last 1/3 of the history
                    for r in range(rec - 1, rec - rec//3 - 2, -1):
                        d = abs(dlmd[ix + i, r])
                        if d == 0:
                            break
                        k = k + 1
                        s = s + d
                    if k < 2 or s == 0:
                        continue
                    # estimate asymptotic convergence factor (a.c.f)
                    qi = abs(dlmd[ix + i, rec - 1])/s
                    if qi <= 0:
                        continue
                    qi = qi**(1.0/(k - 1))
                    acf[1, ix + i] = acf[0, ix + i]
                    acf[0, ix + i] = qi # a.c.f. estimate
                    if qi >= 1.0:
                        continue
                    # esimate error based on a.c.f.
                    theta = qi/(1 - qi)
                    d = theta*dlmd[ix + i, rec - 1]
                    err_lmd[0, ix + i] = abs(d)
                    qx = math.sqrt(qi)
                    err_X[0, ix + i] = dX[ix + i]*qx/(1 - qx)

            if not gen:
                # residual-based error estimates:
                # asymptotic Lehmann for eigenvalues
                # generalized (extended gap) Davis-Kahan for eigenvectors;
                # not valid for the generalized eigenvalue problem
                l = 0
                for k in range(1, leftX):
                    i = ix + k
                    if dX[i] > 0.01:
                        break
                    if lmd[i] - lmd[i - 1] > res[i]:
                        l = k
                if l > 0:
                    i = ix + l
                    t = lmd[i]
                    if verb > 2:
                        print('using left pole at lmd[%d] = %e' % (i, t))
                    m = block_size
                    for k in range(l):
                        i = ix + k
                        s = res[i]
                        err_lmd[1, i] = s*s/(t - lmd[i])
                        err_X[1, i] = s/(t - lmd[i])
                l = 0
                for k in range(1, rightX):
                    i = ix + nx - k - 1
                    if dX[i] > 0.01:
                        break
                    if lmd[i + 1] - lmd[i] > res[i]:
                        l = k
                if l > 0:
                    i = ix + nx - l - 1
                    t = lmd[i]
                    if verb > 2:
                        print('using right pole at lmd[%d] = %e' % (i, t))
                    m = block_size
                    for k in range(l):
                        i = ix + nx - k - 1
                        s = res[i]
                        err_lmd[1, i] = s*s/(lmd[i] - t)
                        err_X[1, i] = s/(lmd[i] - t)

            if verb > 1:
                msg = '  eigenvalue   residual   ' + \
                'estimated errors (kinematic/residual)' + \
                '      a.c.f.'
                print(msg)
                msg = '                          ' + \
                '   eigenvalue            eigenvector '
                print(msg)
                for i in range(block_size):
                    print('%14e %8.1e  %8.1e / %8.1e    %.1e / %.1e  %.3e  %d' % \
                          (lmd[i], res[i], \
                          err_lmd[0, i], err_lmd[1, i], \
                          abs(err_X[0, i]), abs(err_X[1, i]), \
                          acf[0, i], self.cnv[i]))

            eps = epsilon**0.67
            lbs = left_block_size
            if lbs > 0:
                dlmd_min_lft = eps*numpy.amax(abs(dlmd[:lbs, rec - 1]))
            if lbs < block_size:
                dlmd_min_rgt = eps*numpy.amax(abs(dlmd[lbs:, rec - 1]))
            if self.iteration == 2:
                dlmd_min_left = dlmd_min_lft
                dlmd_min_right =  dlmd_min_rgt

            if self.iteration >= 2:
                cluster[:, :] = 0
                nc = 0
                for i in range(left_block_size - 1):
                    if abs(lmd[i + 1] - lmd[i]) <= dlmd_min_lft:
                        if cluster[0, i] == 0:
                            nc += 1
                            cluster[0, i] = nc
                            cluster[1, i] = 1
                        cluster[0, i + 1] = cluster[0, i]
                        cluster[1, i + 1] = cluster[1, i] + 1
                for j in range(m - left_block_size - 1):
                    i = m - j - 1
                    if abs(lmd[i - 1] - lmd[i]) <= dlmd_min_rgt:
                        if cluster[0, i] == 0:
                            nc += 1
                            cluster[0, i] = nc
                            cluster[1, i] = 1
                        cluster[0, i - 1] = cluster[0, i]
                        cluster[1, i - 1] = cluster[1, i] + 1
                if verb > 2:
                    print(cluster[0, :])
                    print(cluster[1, :])

            lcon = 0
            for i in range(leftX - leftX//4):
                if left == 0:
                    break
                j = self.lcon + i
                k = ix + i
                if sigma is not None and lmd[k] > 0:
                    break
                it = iterations[k]
                if it < min_iter:
                    break
                dlmd1 = abs(dlmd[k, max(0, rec - 1)])
                dlmd2 = abs(dlmd[k, max(0, rec - 3)])
                if convergence_criteria.satisfied(self, k):
                    if verb > 0:
                        msg = 'left eigenpair %d converged' + \
                        ' after %d iterations,\n' + \
                        ' eigenvalue %e, error %.1e / %.1e'
                        it = iterations[k]
                        print(msg % (j, it, lmd[k], err_X[0, k], err_X[1, k]))
                    lcon += 1
                    self.cnv[k] = self.iteration + 1
                elif detect_stagn and it > 2 and dlmd1 <= dlmd_min_left \
                     and (dlmd1 > dlmd2 or dlmd1 == 0.0):
                    if verb > 0:
                        msg = 'left eigenpair %d stagnated,\n' + \
                        ' eigenvalue %e, error %.1e / %.1e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    lcon += 1
                    self.cnv[k] = -self.iteration - 1
                else:
                    if cluster[0, k] > 0:
                        for l in range(k - 1, k - cluster[1, k], -1):
                            if self.cnv[l] == -self.iteration - 1:
                                self.cnv[l] = 0
                                lcon -= 1
                                if verb > 0:
                                    msg = 'stagnation of %e cancelled'
                                    print(msg % lmd[l])
                    break

            rcon = 0
            for i in range(rightX - rightX//4):
                if right == 0:
                    break
                j = self.rcon + i
                k = ix + nx - i - 1
                if sigma is not None and lmd[k] < 0:
                    break
                it = iterations[k]
                if it < min_iter:
                    break
                dlmd1 = abs(dlmd[k, max(0, rec - 1)])
                dlmd2 = abs(dlmd[k, max(0, rec - 3)])
                if convergence_criteria.satisfied(self, k):
                    if verb > 0:
                        msg = 'right eigenpair %d converged' + \
                        ' after %d iterations,\n' + \
                        ' eigenvalue %e, residual %.1e, error %.1e / %.1e'
                        print(msg % (j, it, lmd[k], res[k], err_X[0, k], err_X[1, k]))
                    rcon += 1
                    self.cnv[k] = self.iteration + 1
                elif detect_stagn and it > 2 and dlmd1 <= dlmd_min_right \
                     and (dlmd1 > dlmd2 or dlmd1 == 0.0):
                    if verb > 0:
                        msg = 'right eigenpair %d stagnated,\n' + \
                        ' eigenvalue %e, error %.1e / %.1e'
                        print(msg % (j, lmd[k], err_X[0, k], err_X[1, k]))
                    rcon += 1
                    self.cnv[k] = -self.iteration - 1
                else:
                    if cluster[0, k] > 0:
                        for l in range(k + 1, k + cluster[1, k]):
                            if self.cnv[l] == -self.iteration - 1:
                                self.cnv[l] = 0
                                rcon -= 1
                                if verb > 0:
                                    msg = 'stagnation of %e cancelled'
                                    print(msg % lmd[l])
                    break

            if largest: # ensure the largest converge first
                if lcon > 0:
                    i = ix + lcon - 1
                    j = ix + nx - rcon - 1
                    while lcon > 0 and abs(lmd[i]) < abs(lmd[j]):
                        self.cnv[i] = 0
                        lcon -= 1
                        i -= 1
                if rcon > 0:
                    i = ix + lcon
                    j = ix + nx - rcon
                    while rcon > 0 and abs(lmd[i]) > abs(lmd[j]):
                        self.cnv[j] = 0
                        rcon -= 1
                        j += 1

            # move converged X to Xc, update Gram matrix for Xc
            ncon = Xc.nvec()
            if lcon > 0:
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[ix : ix + lcon]))
                self.eigenvalue_errors.append(err_lmd[:, ix : ix + lcon])
                self.eigenvector_errors.append(err_X[:, ix : ix + lcon])
                self.residual_norms = numpy.concatenate \
                    ((self.residual_norms, res[ix : ix + lcon]))
                self.convergence_status = numpy.concatenate \
                    ((self.convergence_status, self.cnv[ix : ix + lcon]))
                X.select(lcon, ix)
                if std and ncon > 0:
                    if ncon > 0:
                        Gu = X.dot(Xc)
                Xc.append(X)
                if not std:
                    if ncon > 0:
                        Gu = X.dot(BXc)
                    BX.select(lcon, ix)
                    BXc.append(BX)
                    if ncon < 1:
                        Gc = BXc.dot(Xc)
                    else:
                        Gl = BXc.dot(X)
                else:
                    if ncon < 1:
                        Gc = Xc.dot(Xc)
                    else:
                        Gl = Xc.dot(X)
                if ncon > 0:
                    Gc = numpy.concatenate((Gc, Gu), axis=1)
                    Gc = numpy.concatenate((Gc, Gl))
                ncon += lcon
            if rcon > 0:
                jx = ix + nx
                self.eigenvalues = numpy.concatenate \
                    ((self.eigenvalues, lmd[jx - rcon : jx]))
                self.eigenvalue_errors.append(err_lmd[:, jx - rcon : jx])
                self.eigenvector_errors.append(err_X[:, jx - rcon : jx])
                self.residual_norms = numpy.concatenate \
                    ((self.residual_norms, res[jx - rcon : jx]))
                self.convergence_status = numpy.concatenate \
                    ((self.convergence_status, self.cnv[jx - rcon : jx]))
                X.select(rcon, jx - rcon)
                if std and ncon > 0:
                    if ncon > 0:
                        Gu = X.dot(Xc)
                Xc.append(X)
                if not std:
                    if ncon > 0:
                        Gu = X.dot(BXc)
                    BX.select(rcon, jx - rcon)
                    BXc.append(BX)
                    if ncon < 1:
                        Gc = BXc.dot(Xc)
                    else:
                        Gl = BXc.dot(X)
                else:
                    if ncon < 1:
                        Gc = Xc.dot(Xc)
                    else:
                        Gl = Xc.dot(X)
                if ncon > 0:
                    Gc = numpy.concatenate((Gc, Gu), axis=1)
                    Gc = numpy.concatenate((Gc, Gl))
                ncon += rcon
            if ncon > 0:
                H = Gc - numpy.identity(ncon, dtype=data_type)
                if verb > 2:
                    print('Gram error: %e' % nla.norm(H))
                # approximate inverse, good enough if Gram offdiagonal
                # entries are less than sqrt(epsilon)
                Gci = 2*numpy.identity(ncon, dtype=data_type) - Gc

            self.lcon += lcon
            self.rcon += rcon
            if options.stopping_criteria is not None:
                if options.stopping_criteria.satisfied(self):
                    return 0
            if largest and right > 0 and self.lcon + self.rcon >= right:
                return 0
            left_converged = left >= 0 and self.lcon >= left
            right_converged = right >= 0 and self.rcon >= right
            if left_converged and right_converged:
                return 0
            if sigma is not None:
                if right_converged:
                    i = ix + lcon
                    lmd_i = lmd[i]
                    err_i = err_lmd[0, i]
                    if lmd_i > 0 and err_i != -1.0 and err_i < lmd_i/4:
                        return 4
                if left_converged:
                    i = ix + nx - rcon - 1
                    lmd_i = lmd[i]
                    err_i = err_lmd[0, i]
                    if lmd_i < 0 and err_i != -1.0 and err_i < -lmd_i/4:
                        return 5
            lim = options.max_quota * eigenvectors.dimension()
            if eigenvectors.nvec() > lim:
                return 1

            leftX -= lcon
            rightX -= rcon

            # re-select Xs, AXs, BXs accordingly
            iy = ix
            ny = nx
            ix += lcon
            nx -= lcon + rcon
            X.select(nx, ix)
            AX.select(nx, ix)
            if not std:
                BX.select(nx, ix)
            XAX = XAX[lcon : lcon + nx, lcon : lcon + nx]
            XBX = XBX[lcon : lcon + nx, lcon : lcon + nx]

            if not pro:
                if P is None:
                    W.copy(Y)
                else:
                    P(W, Y)

            if nz > 0:
                # compute the conjugation matrix
                if pro:
                    ZAY = W.dot(AZ)
                else:
                    ZAY = Y.dot(AZ)
                if std:
                    ZBY = Y.dot(Z)
                else:
                    ZBY = Y.dot(BZ)
                Num = ZAY - numpy.dot(ZBY, numpy.diag(lmd[iy : iy + ny]))
                ny = Y.nvec()
                Lmd = numpy.ndarray((1, ny))
                Mu = numpy.ndarray((nz, 1))
                Lmd[0, :] = lmd[iy : iy + ny]
                Mu[:, 0] = lmdz
                Den = Mu - Lmd
                sy = numpy.sqrt(abs(Y.dots(Y)))
                sz = numpy.sqrt(abs(Z.dots(Z)))
                Beta = numpy.ndarray((nz, ny), dtype=data_type)
                for iz in range(nz):
                    for iy in range(ny):
                        s = sy[iy]/sz[iz]
                        if abs(Num[iz, iy]) >= 100*s*abs(Den[iz, iy]):
                            Beta[iz, iy] = 0.0
                        else:
                            Beta[iz, iy] = Num[iz, iy]/Den[iz, iy]

                # conjugate search directions
                AZ.select(ny)
                Y.add(Z, -1.0, Beta)
                if pro: # if gen or nz == 0, BY computed later
                    W.add(BZ, -1.0, Beta)
                    BY.select(ny)
                    W.copy(BY)
            elif pro:
                BY.select(ny)
                W.copy(BY)

            Q = Y.dot(BX)
            Y.add(X, -1.0, Q)
            if pro:
                BY.add(BX, -1.0, Q)

            if Xc.nvec() > 0:
                # orthogonalize Y to Xc
                # std: W := W - Xc Xc* W (not needed if P is None)
                # gen: W := W - Xc BXc* W
                # pro: W := W - Xc BXc* W (not needed if P is None)
                Q = numpy.dot(Gci, Y.dot(BXc))
                Y.add(Xc, -1.0, Q)
                if pro:
                    BY.add(BXc, -1.0, Q)

            # compute (B-)Gram matrix for (X,Y)
            if std:
                s = numpy.sqrt(abs(Y.dots(Y)))
                Y.scale(s)
                if nx > 0:
                    XBY = Y.dot(X)
                YBY = Y.dot(Y)
            else:
                BY.select(Y.nvec())
                if not pro: # or nz == 0:
                    B(Y, BY)
                s = numpy.sqrt(abs(BY.dots(Y)))
                Y.scale(s)
                BY.scale(s)
                if nx > 0:
                    XBY = BY.dot(X)
                YBY = BY.dot(Y)

            if nx > 0:
                YBX = _conjugate(XBY)
                GB = numpy.concatenate((XBX, YBX))
                H = numpy.concatenate((XBY, YBY))
                GB = numpy.concatenate((GB, H), axis=1)
            else:
                GB = YBY

            # do pivoted Cholesky for GB
            U = GB
            ny = Y.nvec()
            if single:
                eps = 1e-3
            else:
                eps = 1e-8
            ind, dropped = _piv_chol(U, nx, eps)
            if dropped > 0:
                if verb > 0:
                    print('dropped %d search directions out of %d' \
                          % (dropped, ny))

            ny -= dropped
            if ny < 1:
                if verb > -1:
                    print('no search directions left, terminating')
                return 3

            # re-arrange/drop-linear-dependent search directions
            nxy = nx + ny
            U = U[:nxy, :nxy]

            indy = ind[nx: nxy]
            for i in range(ny):
                indy[i] -= nx
            W.select(ny)
            Y.copy(W, indy[:ny])
            Y.select(ny)
            W.copy(Y)
            AY.select(ny)
            if not std:
                BY.copy(W, indy[:ny])
                BY.select(ny)
                W.copy(BY)

            # compute A-Gram matrix for (X,Y)
            if pro:
                A(BY, AY)
                if nx > 0:
                    XAY = AY.dot(BX)
                YAY = AY.dot(BY)
            else:
                A(Y, AY)
                if nx > 0:
                    XAY = AY.dot(X)
                YAY = AY.dot(Y)
            if nx > 0:
                YAX = _conjugate(XAY)
                GA = numpy.concatenate((XAX, YAX))
                H = numpy.concatenate((XAY, YAY))
                GA = numpy.concatenate((GA, H), axis=1)
            else:
                GA = YAY

            # solve Rayleigh-Ritz eigenproblem
            G = _transform(GA, U)
            YAY = G[nx : nxy, nx : nxy]
            lmdy, Qy = sla.eigh(YAY)
            G[:, nx : nxy] = numpy.dot(G[:, nx : nxy], Qy)
            if nx > 0:
                G[nx : nxy, :nx] = _conjugate(G[:nx, nx : nxy])
            G[nx : nxy, nx : nxy] = numpy.dot(_conjugate(Qy), G[nx : nxy, nx : nxy])

            if G.dtype.kind == 'c':
                G = G.astype(numpy.complex128)
            else:
                G = G.astype(numpy.float64)

            lmdxy, Q = sla.eigh(G, turbo = False)

            lmdxy = lmdxy.astype(lmdy.dtype)
            Q = Q.astype(Qy.dtype)

            # estimate changes in eigenvalues and eigenvectors
            lmdx = numpy.concatenate \
                ((lmdxy[:leftX], lmdxy[nxy - rightX:]))
            lmdy = lmdxy[leftX : nxy - rightX]
            QX = numpy.concatenate \
                ((Q[:, :leftX], Q[:, nxy - rightX:]), axis=1)
            QYX = QX[nx:, :].copy()
            lmdX = numpy.ndarray((1, nx))
            lmdY = numpy.ndarray((ny, 1))
            lmdX[0, :] = lmdx
            lmdY[:, 0] = lmdy
            Delta = (lmdY - lmdX)*QYX*QYX
            dX[ix : ix + nx] = _norm(QYX, 0)
            if rec == RECORDS:
                for i in range(rec - 1):
                    dlmd[:, i] = dlmd[:, i + 1]
            else:
                rec += 1
            dlmd[ix : ix + nx, rec - 1] = _real(numpy.sum(Delta, axis=0))

            # select new numbers of left and right eigenpairs
            if left < 0:
                shift_left = ix
            elif lcon > 0:
                shift_left = max(0, left_total - self.lcon - leftX)
                shift_left = min(shift_left, ix)
            else:
                shift_left = 0
            if right < 0:
                shift_right = block_size - ix - nx
            elif rcon > 0:
                shift_right = max(0, right_total - self.rcon - rightX)
                shift_right = min(shift_right, block_size - ix - nx)
            else:
                shift_right = 0
            if shift_left + shift_right > ny:
                shift_left = min(shift_left, int(round(left_ratio*ny)))
                shift_right = min(shift_right, ny - shift_left)
            if left > 0 and lcon > 0 and self.lcon >= left:
                if verb > 0:
                    print('left-hand side converged')
                leftX_new = 0
                l = left_block_size
                rightX_new = min(nxy, l + rightX + shift_right)
                left_block_size_new = l + rightX + shift_right - rightX_new
                shift_left = -leftX - lcon
                left_ratio = 0.0
                ix_new = left_block_size_new
            elif right > 0 and rcon > 0 and self.rcon >= right:
                if verb > 0:
                    print('right-hand side converged')
                ix_new = ix - shift_left
                leftX_new = min(nxy, block_size - ix_new)
                rightX_new = 0
                shift_right = -rightX - rcon
                left_block_size_new = ix_new + leftX_new
                left_ratio = 1.0
            else:
                leftX_new = leftX + shift_left
                rightX_new = rightX + shift_right
                left_block_size_new = left_block_size
                ix_new = ix - shift_left
            nx_new = leftX_new + rightX_new
            if verb > 2:
                print('left X: was %d, now %d' % (leftX, leftX_new))
                print('right X: was %d, now %d' % (rightX, rightX_new))
                print('new ix %d, new nx %d, nxy %d' % (ix_new, nx_new, nxy))

            # shift eigenvalues etc.
            m = block_size
            l = left_block_size
            nl = left_block_size_new
            cnv = self.cnv
            if shift_left > 0:
                for i in range(l - shift_left):
                    j = i + shift_left
                    cnv[i] = cnv[j]
                    lmd[i] = lmd[j]
                    res[i] = res[j]
                    acf[:, i] = acf[:, j]
                    err_lmd[:, i] = err_lmd[:, j]
                    dlmd[i, :] = dlmd[j, :]
                    err_X[:, i] = err_X[:, j]
                    dX[i] = dX[j]
                    iterations[i] = iterations[j]
            if shift_left >= 0:
                for i in range(l - shift_left, nl):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)
            else:
                for i in range(l):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)
            if shift_right > 0:
                for i in range(m - 1, l + shift_right - 1, -1):
                    j = i - shift_right
                    cnv[i] = cnv[j]
                    lmd[i] = lmd[j]
                    res[i] = res[j]
                    acf[:, i] = acf[:, j]
                    err_lmd[:, i] = err_lmd[:, j]
                    dlmd[i, :] = dlmd[j, :]
                    err_X[:, i] = err_X[:, j]
                    dX[i] = dX[j]
                    iterations[i] = iterations[j]
            if shift_right >= 0:
                for i in range(l + shift_right - 1, nl - 1, -1):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)
            else:
                for i in range(l, block_size):
                    _reset_cnv_data \
                        (i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations)

            # compute RR coefficients for X and 'old search directions' Z
            # by re-arranging columns of Q
            Q[nx : nxy, :] = numpy.dot(Qy, Q[nx : nxy, :])
            Q = sla.solve_triangular(U, Q)
            lmdx = numpy.concatenate \
                ((lmdxy[:leftX_new], lmdxy[nxy - rightX_new:]))
            QX = numpy.concatenate \
                ((Q[:, :leftX_new], Q[:, nxy - rightX_new:]), axis=1)
            lft = leftX_new
            rgt = rightX_new
            nz = nxy - lft - rgt
            lmdz = lmdxy[lft : nxy - rgt]
            QZ = Q[:, lft : nxy - rgt]
            if nx > 0:
                QXX = QX[:nx, :].copy()
            QYX = QX[nx:, :].copy()
            if nx > 0:
                QXZ = QZ[:nx, :].copy()
            QYZ = QZ[nx:, :].copy()

            # update X and 'old search directions' Z and their A- and B-images
            W.select(nx_new)
            Z.select(nx_new)
            if nx > 0:
                AX.multiply(QXX, W)
                W.add(AY, 1.0, QYX)
            else:
                AY.multiply(QYX, W)
            if nz > 0:
                Z.select(nz)
                AY.multiply(QYZ, Z)
                AZ.select(nz)
                if nx > 0:
                    Z.add(AX, 1.0, QXZ)
                Z.copy(AZ)
            AX.select(nx_new, ix_new)
            W.copy(AX)
            if not std:
                Z.select(nx_new)
                if nx > 0:
                    BX.multiply(QXX, W)
                    W.add(BY, 1.0, QYX)
                else:
                    BY.multiply(QYX, W)
                if nz > 0:
                    Z.select(nz)
                    BY.multiply(QYZ, Z)
                    BZ.select(nz)
                    if nx > 0:
                        Z.add(BX, 1.0, QXZ)
                    Z.copy(BZ)
                BX.select(nx_new, ix_new)
                W.copy(BX)
            else:
                BZ = Z
            Z.select(nx_new)
            if nx > 0:
                X.multiply(QXX, W)
                W.add(Y, 1.0, QYX)
            else:
                Y.multiply(QYX, W)
            if nz > 0:
                Z.select(nz)
                Y.multiply(QYZ, Z)
                if nx > 0:
                    Z.add(X, 1.0, QXZ)
            X.select(nx_new, ix_new)
            W.copy(X)

            nx = nx_new
            ix = ix_new
            leftX = leftX_new
            rightX = rightX_new
            left_block_size = left_block_size_new
            self.iteration += 1

        return 2

class _Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return '??? ' + repr(self.value)

def _conjugate(a):
    if a.dtype.kind == 'c':
        return a.conj().T
    else:
        return a.T

def _real(a):
    if a.dtype.kind == 'c':
        return a.real
    else:
        return a

def _transform(A, U):
    B = sla.solve_triangular(_conjugate(U), _conjugate(A), lower=True)
    A = sla.solve_triangular(_conjugate(U), _conjugate(B), lower=True)
    return A

def _default_block_size(left, right, extra, init, threads):
    extra_left = int(extra[0])
    extra_right = int(extra[1])
    init_left = 0
    init_right = 0
    if init[0] is not None:
        init_left = int(init[0].nvec())
    if init[1] is not None:
        init_right = int(init[1].nvec())
    if threads <= 8:
        threads = 8
    if left == 0 and right == 0:
        return 0
    if left <= 0 and right <= 0:
        if init_left == 0 and init_right == 0:
            if left < 0 and right < 0:
                return 2*threads
            else:
                return threads
        m = init_left + init_right
        m = threads*((m - 1)//threads + 1)
        if left < 0 or right < 0:
            m = max(m, 2*threads)
        return m
    left_total = 0
    right_total = 0
    if left > 0:
        if extra_left >= 0:
            left_total = max(left + extra_left, init_left)
        else:
            left_total = int(math.floor(max(left, init_left)*1.2))
    if right > 0:
        if extra_right >= 0:
            right_total = max(right + extra_right, init_right)
        else:
            right_total = int(math.floor(max(right, init_right)*1.2))
    if left < 0:
        left_total = right_total
    if right < 0:
        right_total = left_total
    m = int(left_total + right_total)
    m = threads*((m - 1)//threads + 1)
    if left < 0 or right < 0:
        m = max(m, 2*threads)
    return m

def _reset_cnv_data(i, cnv, res, acf, err_lmd, dlmd, err_X, dX, iterations):
    cnv[i] = 0
    res[i] = -1.0
    acf[:, i] = 1.0
    err_lmd[:, i] = -1.0
    dlmd[i, :] = 0
    err_X[:, i] = -1.0
    dX[i] = 1.0
    iterations[i] = 0

def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)

def _piv_chol(A, k, eps, blk=64, verb=0):
    n = A.shape[0]
    buff = A[0, :].copy()
    ind = [i for i in range(n)]
    drop_case = 0
    dropped = 0
    last_check = -1
    if k > 0:
        U = sla.cholesky(A[:k, :k])
        A[:k, :k] = U.copy()
        A[:k, k : n] = sla.solve_triangular \
                       (_conjugate(U), A[:k, k : n], lower=True)
        A[k : n, :k].fill(0.0)
        A[k : n, k : n] -= numpy.dot(_conjugate(A[:k, k : n]), A[:k, k : n])
    l = k
    for i in range(k, n):
        s = numpy.diag(A[i : n, i : n]).copy()
        if i > l:
            t = _norm(A[l : i, i : n], 0)
            s -= t*t
        j = i + numpy.argmax(s)
        if i != j:
            buff[:] = A[i, :]
            A[i, :] = A[j, :]
            A[j, :] = buff
            buff[:] = A[:, i]
            A[:, i] = A[:, j]
            A[:, j] = buff
            ind[i], ind[j] = ind[j], ind[i]
        if i > l:
            A[i, i : n] -= numpy.dot(_conjugate(A[l : i, i]), A[l : i, i : n])
        last_piv = A[i, i].real
        if last_piv <= eps:
            A[i : n, :].fill(0.0)
            drop_case = 1
            dropped = n - i
            break
        A[i, i] = math.sqrt(last_piv)
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
        if i - l == blk - 1 or i == n - 1:
            last_check = i
            lmin = _estimate_lmin(A[: i + 1, : i + 1])
            lmax = _estimate_lmax(A[: i + 1, : i + 1])
            if verb > 0:
                print('%e %e %e' % (A[i, i], lmin, lmax))
            if lmin/lmax <= eps:
                A[i : n, :].fill(0.0)
                drop_case = 2
                dropped = n - i
                break
        if i - l == blk - 1 and i < n - 1:
            j = i + 1
            A[j : n, j : n] -= numpy.dot(_conjugate(A[l : j, j : n]), \
                A[l : j, j : n])
            l += blk
    if last_check < n - 1 and drop_case != 2:
        i = last_check
        j = n - dropped - 1
        while i < j:
            m = i + (j - i + 1)//2
            lmin = _estimate_lmin(A[: m + 1, : m + 1])
            lmax = _estimate_lmax(A[: m + 1, : m + 1])
            if verb > 0:
                print('%d %e %e' % (m, lmin, lmax))
            if lmin/lmax <= eps:
                if j > m:
                    j = m
                    continue
                else:
                    A[j : n, :].fill(0.0)
                    dropped = n - j
                    last_piv = A[j - 1, j - 1]**2
                    break
            else:
                i = m
                continue
    return ind, dropped

def _estimate_lmax(U):
    U = numpy.triu(U)
    return sla.norm(numpy.dot(_conjugate(U), U), ord=1)
def _estimate_lmin(U):
    n = U.shape[0]
    if U.dtype.kind == 'c':
        tr = 2
    else:
        tr = 1
    x = numpy.ones((n,), dtype=U.dtype)
    s = numpy.dot(x, x)
    for i in range(3):
        y = sla.solve_triangular(U, x, trans=tr)
        t = numpy.dot(y, y)
        rq = s/t
        x = sla.solve_triangular(U, y)
        s = numpy.dot(x, x)
    return rq
