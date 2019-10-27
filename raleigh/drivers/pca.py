# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""Principal Component Analysis of a dataset represented by a 2D ndarray.
"""

import math
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
    that the product L R approximates A - e a, where e = numpy.ones((m, 1)),
    and a = numpy.mean(A, axis=0).reshape((1, n))

    The rows of R (principal components) are orhonormal, the columns of L
    (reduced-features data) are in the descending order of their norms.

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
        D = A - e a - L R becomes not greater than eps, where eps is the norm
        of A multiplied by tol if tol > 0 and eps = -tol if tol < 0. Otherwise
        the user will be asked repeatedly whether the computation should 
        continue (the number of computed principal components and the relative 
        truncation error, the ratio of the norm of D to that of A, are 
        displayed).
    have : tuple (a0, L0, R0)
        If not None, previously computed PCA approximation L0 R0 + e0 a0
        to a data matrix A0 of the same width as A is to be updated, i.e.
        PCA approximation L R + e a to numpy.concatenate((A0, A)) is to be
        computed.
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
        The norm to be used for evaluating the approximation error:
        's' : the largest singular value of D (if tol > 0, then the largest
              singular value of A - e a will be used instead of the norm of A
              for evaluating the relative truncation error),
        'f' : Frobenius norm of D,
        'm' : the largest norm of a row of D.
    mpc : int
        Maximal number of PCs to compute. Ignored if negative, otherwise
        if mpc < min(m, n), then the required accuracy of approximation
        might not be achieved.
    svtol : float
        Error tolerance for singular values (see Notes below) relative to
        the largest singular value.
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
                        max_rank=mpc, svtol=svtol, shift=True, arch=arch, \
                        verb=verb)
    trans = lra.left() # transfomed (reduced-features) data
    comps = lra.right() # principal components
    return lra.mean(), trans, comps


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