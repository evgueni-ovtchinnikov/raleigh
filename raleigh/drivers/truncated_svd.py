# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

# -*- coding: utf-8 -*-
"""Truncated Singular Value Decomposition of a matrix 
   represented by a 2D ndarray.
"""

from ..core.solver import Options
from .partial_svd import AMatrix
from .partial_svd import PartialSVD
from .partial_svd import DefaultStoppingCriteria


def truncated_svd(matrix, opt=Options(), nsv=-1, tol=-1, norm='s', msv=-1, \
                  vtol=1e-3, arch='cpu', verb=0):
    '''Computes truncated Singular Value Decomposition of a dense matrix A.
    
    For a given m by n matrix A computes m by k matrix U, k by k diagonal
    matrix S and n by k matrix V such that A V = U S, the columns of U and V
    are orthonirmal (orthogonal and of unit norm) and the largest singular
    value of A - U S V' is smallest possible for a given k (V' = V.T for a real
    A and A.T.conj() for a complex A).
    The diagonal entries of S are the largest k singular values of A and the
    columns of U and V are corresponding left and right singular vectors.

    Parameters
    ----------
    matrix : 2D numpy array
        Matrix A.
    opt : an object of class raleigh.solver.Options
        Solver options (see raleigh.solver).
    nsv : int
        Required number of singular values and vectors if known.
        If negative, implicitely defined by the required truncation tolerance
        or interactively by the user.
    tol : float
        Truncation tolerance in the case nsv < 0. If tol is non-zero, then the 
        computation of singular values and vectors will stop when the norm of 
        D = A - U S V' becomes not greater than eps, where eps is the norm of 
        A multiplied by tol if tol > 0 and eps = -tol if tol < 0. If tol is 
        zero, then the user will be asked repeatedly whether the computation
        should continue (the number of computed singular values and the 
        relative truncation error, the ratio of the norm of D to that of A, are
        displayed).
    norm : character
        The norm to be used for evaluating the truncation error:
        's' : the largest singular value of D,
        'f' : Frobenius norm of D,
        'm' : the largest norm of a row of D.
    msv : int
        Maximal number of singular values to compute. Ignored if negative, 
        otherwise if msv < min(m, n), then the truncation error may be greater
        than requested.
    vtol : float
        Singular vectors error tolerance.
    arch : string
        'cpu' : run on CPU,
        'gpu' : run on GPU if available, otherwise on CPU,
        'gpu!' : run on GPU, throw RuntimError if GPU is not present.
    verb : int
        Verbosity level.

    Returns
    -------
    u : numpy array of shape (m, k)
        The matrix U.
    sigma : numpy array of shape (k,)
        The array of the largest k singular values in descending order.
    vt : numpy array of shape (k, n)
        The matrix V'.

    Notes
    -----
    This solver can only be efficient if singular values decay quickly, e.g.
    exponentially. If the singular values properties are unknown, then it is
    worth to try it first in interactive mode (nsv < 0, tol = 0), and if the
    computation is too slow, use scipy.linalg.svd instead.
    '''
    matrix = AMatrix(matrix, arch=arch)
    user_bs = opt.block_size
    if user_bs < 1 and (nsv < 0 or nsv > 100):
        opt.block_size = 128
    if opt.convergence_criteria is None:
        no_cc = True
        opt.convergence_criteria = _DefaultSVDConvergenceCriteria(vtol)
    else:
        no_cc = False
    if opt.stopping_criteria is None and nsv < 0:
        no_sc = True
        opt.stopping_criteria = \
            DefaultStoppingCriteria(matrix, tol, norm, msv, verb)
    else:
        no_sc = False

    psvd = PartialSVD()
    psvd.compute(matrix, opt, nsv=(0, nsv))
    u = psvd.left()
    v = psvd.right()
    sigma = psvd.sigma
    if msv > 0 and u.shape[1] > msv:
        u = u[:, : msv]
        v = v[:, : msv]
        sigma = sigma[: msv]

    # restore user opt to avoid side effects
    opt.block_size = user_bs
    if no_cc:
        opt.convergence_criteria = None
    if no_sc:
        opt.stopping_criteria = None

    return u, sigma, v.T


class _DefaultSVDConvergenceCriteria:
    def __init__(self, tol):
        self.tolerance = tol
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def satisfied(self, solver, i):
        err = solver.convergence_data('kinematic vector error', i)
        return err >= 0 and err <= self.tolerance