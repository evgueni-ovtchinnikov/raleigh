import numpy
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
    return vals, vecs, status

def py_pca(data, opts):
    npc = -1
    tol = 0.01
    NORMS = ['s', 'f', 'm']
    inorm = 0
    if ~_isempty(opts):
        if 'tolerance' in opts:
            tol = opts['tolerance']
        if 'num_pc' in opts:
            npc = opts['num_pc']
        if 'norm_err' in opts:
            inorm = opts['norm_err']
    data = numpy.asarray(data)
    mean, trans, comps = pca(data.T, npc=npc, tol=tol, norm=NORMS[inorm])
    return mean, trans, comps
