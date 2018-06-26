'''Pivoted Cholesky for a matrix with unit diagonal
'''

import math
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

def conjugate(a):
    if a.dtype.kind == 'c':
#    if np.iscomplex(a).any():
        return a.conj().T
    else:
        return a.T

def piv_chol(A, k, eps, verb = 0):
    n = A.shape[0]
    ind = [i for i in range(n)]
    if k > 0:
        U = sla.cholesky(A[:k,:k])
        A[:k,:k] = U.copy()
        A[:k, k : n] = sla.solve_triangular \
                       (conjugate(U), A[:k, k : n], lower = True)
        A[k : n, :k].fill(0.0)
    for i in range(k, n):
        if i > 0:
            s = nla.norm(A[:i, i : n], axis = 0)
            j = i + np.argmin(s)
            if i != j:
                #print('swapping %d and %d' % (i, j))
                A[i, :], A[j, :] = A[j, :], A[i, :].copy()
                A[:, i], A[:, j] = A[:, j], A[:, i].copy()
                ind[i], ind[j] = ind[j], ind[i]
            A[i, i : n] -= np.dot(conjugate(A[:i, i]), A[:i, i : n])
        last_piv = A[i, i]
        #print('pivot: %e' % last_piv)
        if A[i, i] <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, last_piv
        A[i, i] = math.sqrt(abs(A[i, i]))
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
        lmin = estimate_lmin(A[: i + 1, : i + 1])
        lmax = estimate_lmax(A[: i + 1, : i + 1])
        if verb > 0:
            print('%e %e %e' % (A[i,i], lmin, lmax))
        if lmin/lmax <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, last_piv
    return ind, 0, A[n - 1, n - 1]**2

def estimate_lmax(U):
    U = np.triu(U)
    return sla.norm(np.dot(conjugate(U), U), ord = 1)
def estimate_lmin(U):
    n = U.shape[0]
    if U.dtype.kind == 'c':
        tr = 2
    else:
        tr = 1
    x = np.ones((n,), dtype = U.dtype)
    s = np.dot(x, x)
    for i in range(3):
        y = sla.solve_triangular(U, x, trans = tr)
        t = np.dot(y, y)
        rq = s/t
        #print(i, rq)
        x = sla.solve_triangular(U, y)
        s = np.dot(x, x)
    return rq
