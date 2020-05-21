# Copyright 2019 United Kingdom Research and Innovation
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

'''MATLAB interface (WIP, tested with Python3.6 only)
'''

import numpy
import numpy.linalg as nla
from scipy.sparse import csr_matrix

from ..algebra import verbosity
verbosity.level = 2
from .partial_hevp import partial_hevp as evp
from .pca import pca


def _isempty(x):
    try:
        info = x.buffer_info()
        return info[1] < 1
    except:
        return False


def pca_error(data, mean, trans, comps):
    m, n = data.shape
    ones = numpy.ones((data.shape[0], 1), dtype=data.dtype)
    mean = numpy.reshape(mean, (1, comps.shape[1]))
    data_s = data - numpy.dot(ones, mean)
    err = numpy.dot(trans, comps) - data_s
    em = numpy.amax(_norm(err, axis=1))/numpy.amax(_norm(data_s, axis=1))
    ef = nla.norm(err, ord='fro')/nla.norm(data_s, ord='fro')
    return em, ef


def _norm(a, axis):
    return numpy.apply_along_axis(nla.norm, axis, a)


def part_hevp(n, rowA, colA, valA, nep, sigma, rowB, colB, valB, opts):
    tol = 1e-4
    verb = 0
    buckling = False
    if ~_isempty(opts):
        if 'tolerance' in opts:
            tol = opts['tolerance']
        if 'verbosity' in opts:
            verb = opts['verbosity']
        if 'buckling' in opts:
            buckling = opts['buckling']
    n = int(n)
    rowA = numpy.asarray(rowA, dtype=numpy.int32) - 1
    colA = numpy.asarray(colA, dtype=numpy.int32) - 1
    valA = numpy.asarray(valA)
    matrixA = csr_matrix((valA, (rowA, colA)), shape=(n,n))
    if _isempty(rowB):
        matrixB = None
    else:
        rowB = numpy.asarray(rowB, dtype=numpy.int32) - 1
        colB = numpy.asarray(colB, dtype=numpy.int32) - 1
        valB = numpy.asarray(valB)
        matrixB = csr_matrix((valB, (rowB, colB)), shape=(n,n))
    vals, vecs, status = evp(matrixA, B=matrixB, sigma=sigma, which=nep, \
                             tol=tol, verb=verb, buckling=buckling)
    if buckling:
        vals = -vals
    k = vals.shape[0]
    return vals, vecs.T.reshape((k*n,)), status
#    return vals, vecs, status

def py_pca(data, m, n, opts): # n samples m features each
    m = int(m)
    n = int(n)
    npc = -1
    tol = 0.01
    NORMS = ['f', 's', 'm']
    inorm = 0
    if ~_isempty(opts):
        if 'tolerance' in opts:
            tol = opts['tolerance']
        if 'num_pc' in opts:
            npc = int(opts['num_pc'])
        if 'norm_err' in opts:
            inorm = int(opts['norm_err'])
    data = numpy.asarray(data).reshape((n, m))
    mean, trans, comps = pca(data, npc=npc, tol=tol, norm=NORMS[inorm])
#    mean, trans, comps = pca(data.T, npc=npc, tol=tol, norm=NORMS[inorm])
#    em, ef = pca_error(data, mean, trans, comps)
#    print(em, ef)
    mean = numpy.reshape(mean, (m,))
    npc = comps.shape[0]
    trans = numpy.reshape(trans, (n*npc,))
    comps = numpy.reshape(comps, (m*npc,))
    return mean, npc, comps, trans
