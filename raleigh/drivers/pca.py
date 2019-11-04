# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""Principal Component Analysis of a dataset represented by a 2D ndarray.
"""

import numpy
import numpy.linalg as nla

from ..core.solver import Options
from .partial_svd import AMatrix
from .lra import LowerRankApproximation


def pca(A, npc=-1, tol=0, have=None, batch_size=None, verb=0, arch='cpu', \
        norm='f', mpc=-1, svtol=1e-3, opt=Options()):
    '''Performs principal component analysis for the set of data items
    represented by rows of a dense matrix A.

    For a given m-by-n data matrix A (m data samples n features each) computes
    m-by-k matrix L and k-by-n matrix R with k not greater than m and n such
    that the product L R approximates A_s = A - e a, where e = numpy.ones((m, 1)),
    and a = numpy.mean(A, axis=0).reshape((1, n))

    The rows of R (principal components) are orhonormal, the columns of L
    (reduced-features data) are orthogonal and are in the descending order of
    their norms.

    Parameters: basic
    -----------------
    A : 2D numpy array
        Data matrix.
    npc : int
        Required number of principal components if known.
        If negative, implicitely defined by the required accuracy of
        PCA approximation or interactively by the user.
    tol : float
        Approximation tolerance in the case npc < 0. If tol is non-zero, then 
        the computation of principal components will stop when the norm of 
        D = A_s - L R becomes not greater than eps, where eps is the norm
        of A_s multiplied by tol if tol > 0 and eps = -tol if tol < 0. Otherwise
        the user will be asked repeatedly whether the computation should 
        continue (the number of computed principal components and the relative 
        truncation error, the ratio of the norm of D to that of A_s, are
        displayed). Ignored if npc is positive.
    mpc : int
        Maximal number of PCs to compute. Ignored if non-positive, otherwise
        if mpc < min(m, n), then the accuracy of approximation set by tol
        might not be achieved. Ignored if npc is positive.
    have : tuple (a0, L0, R0)
        If not None, previously computed PCA approximation L0 R0 + e0 a0
        to a data matrix A0 of the same width as A is to be updated, i.e.
        PCA approximation L R + e a to numpy.concatenate((A0, A)) is to be
        computed. If neither nps nor tol are specified, the same number of
        components as in R0 is to be computed.
    batch_size : int
        If not None, performs incremental PCA with specified batch size.
        Principal components for the first batch are computed by the standard
        SVD, for each of the rest, update is performed (see 'have' above).
    verb : int
        Verbosity level.
    arch : string
        'cpu' : run on CPU,
        'gpu' : run on GPU if available, otherwise on CPU,
        'gpu!' : run on GPU, throw RuntimError if GPU is not present.

    Parameters: advanced
    --------------------
    norm : character
        The measure to be used for evaluating the relative approximation error:
        's' : the largest singular value,
        'f' : Frobenius norm,
        'm' : the largest norm of a row.
    svtol : float
        Error tolerance for singular values (see Notes below) relative to
        the largest singular value. May need to be decreased from its default
        value if highly accurate approximation is sought.
    opt : an object of class raleigh.solver.Options
        Solver options (see raleigh.solver).

    Returns
    -------
    mean : numpy array of shape (1, n)
        The mean of rows of A.
    trans : numpy array of shape (m, k)
        The reduced-features data matrix.
    comps : numpy array of shape (k, n)
        Principal components matrix.

    Usage examples
    --------------
    - Generate test data
    >>> import numpy
    >>> from ..examples.pca.generate_matrix import generate
    >>> numpy.random.seed(1)
    >>> A, sigma, u, v = generate(3000, 2000, 1000, pca=True)

    - Compute 300 principal components of A:
    >>> mean, trans, comps = pca(A, npc=300)
    >>> em, ef = pca_error(A, mean, trans, comps)
    >>> print('PCA error: max 2-norm %.0e, Frobenius norm %.0e' % (em, ef))
    PCA error: max 2-norm 5e-02, Frobenius norm 1e-01

    - Compute a number of principal components delivering 5% relative
      accuracy of approximation of A:
    >>> mean, trans, comps = pca(A, tol=0.05)
    >>> print('%d components computed' % comps.shape[0])
    906 components computed
    >>> em, ef = pca_error(A, mean, trans, comps)
    >>> print('PCA error: max 2-norm %.0e, Frobenius norm %.0e' % (em, ef))
    PCA error: max 2-norm 2e-02, Frobenius norm 4e-02

    - Compute PCA to 5% accuracy incrementally by processing 1000 data
      samples at a time:
    >>> mean, trans, comps = pca(A, batch_size=1000, tol=0.05)
    >>> print('%d components computed' % comps.shape[0])
    926 components computed
    >>> em, ef = pca_error(A, mean, trans, comps)
    >>> print('PCA error: max 2-norm %.0e, Frobenius norm %.0e' % (em, ef))
    PCA error: max 2-norm 2e-02, Frobenius norm 4e-02

    - To demonstrate the use of update, let A[:2400, :] play the role of 'old'
      data A0
    >>> A0 = A[:2400, :]
    >>> mean, trans, comps = pca(A0, tol=0.05)
    >>> print('%d components computed' % comps.shape[0])
    846 components computed
    >>> em, ef = pca_error(A0, mean, trans, comps)
    >>> print('PCA error: max 2-norm %.0e, Frobenius norm %.0e' % (em, ef))
    PCA error: max 2-norm 2e-02, Frobenius norm 5e-02

    - Now 'new' data A1 = A[2400:, :] arrived - update previously computed
      principal components:
    >>> A1 = A[2400:, :]
    >>> mean, trans, comps = pca(A1, have=(mean, trans, comps))
    >>> print('%d components updated' % comps.shape[0])
    846 components updated
    >>> em, ef = pca_error(A, mean, trans, comps)
    >>> print('PCA error: max 2-norm %.0e, Frobenius norm %.0e' % (em, ef))
    PCA error: max 2-norm 2e-02, Frobenius norm 5e-02

    Notes
    -----
    If have is None, then the rows of R are approximate right singular
    vectors of A - e a and the columns of L are approximate left singular
    vectors of A - e a multiplied by respective singular values. If have is
    not None, this is generally not the case.
    '''
    lra = LowerRankApproximation(have)
    if batch_size is None:
        if have is None:
            data_matrix = AMatrix(A, arch=arch)
            lra.compute(data_matrix, opt=opt, rank=npc, tol=tol, norm=norm, \
                        max_rank=mpc, svtol=svtol, shift=True, verb=verb)
        else:
            data_matrix = AMatrix(A, arch=arch, copy_data=True)
            lra.update(data_matrix, opt=opt, rank=npc, tol=tol, norm=norm, \
                       max_rank=mpc, svtol=svtol, verb=verb)
    else:
        lra.icompute(A, batch_size, opt=opt, rank=npc, tol=tol, norm=norm, \
                     max_rank=mpc, svtol=svtol, shift=True, verb=verb, \
                     arch=arch)
    trans = lra.left()
    comps = lra.right()
    mean = lra.mean()
    return mean, trans, comps


def pca_error(data, mean, trans, comps):
    m, n = data.shape
    ones = numpy.ones((data.shape[0], 1), dtype=data.dtype)
    mean = numpy.reshape(mean, (1, comps.shape[1]))
    data_s = data - numpy.dot(ones, mean)
    err = numpy.dot(trans, comps) - data_s
    em = numpy.amax(_norm(err, axis=1))/numpy.amax(_norm(data_s, axis=1))
    ef = numpy.amax(nla.norm(err, ord='fro')) \
        /numpy.amax(nla.norm(data_s, ord='fro'))
    return em, ef


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)
