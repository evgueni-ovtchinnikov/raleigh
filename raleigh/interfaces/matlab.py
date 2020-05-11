import numpy
from scipy.sparse import csr_matrix

from ..algebra import verbosity
verbosity.level = 2
from .partial_hevp import partial_hevp as evp

def _isempty(x):
    try:
        info = x.buffer_info()
        return info[1] < 1
    except:
        return False

def partial_hevp(n, rowA, colA, valA, nep, sigma, rowB, colB, valB, opts):
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
