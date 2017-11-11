import math
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

def piv_chol(A, k, eps):
    lmax = sla.norm(A, ord = 1)
    n = A.shape[0]
    ind = [i for i in range(n)]
    if k > 0:
        U = sla.cholesky(A[:k,:k])
        A[:k,:k] = U.copy()
        A[:k, k : n] = sla.solve_triangular \
                       (U.transpose(), A[:k, k : n], lower = True)
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
            A[i, i : n] -= np.dot(A[:i, i].transpose(), A[:i, i : n])
        #print('pivot: %e' % A[i,i])
        if A[i, i] <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, A[i, i]
        A[i, i] = math.sqrt(A[i, i])
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
        lmin = estimate_lmin(A[: i + 1, : i + 1])
        #print('%e %e %e' % (A[i,i], lmin, lmax))
        if lmin/lmax <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, A[i, i]
    return ind, 0, A[n - 1, n - 1]**2

def estimate_lmin(U):
    n = U.shape[0]
    x = np.ones((n,), dtype = np.dtype(U[0,0]))
    s = np.dot(x, x)
    for i in range(3):
        y = sla.solve_triangular(U, x, trans = 2)
        t = np.dot(y, y)
        rq = s/t
        #print(i, rq)
        x = sla.solve_triangular(U, y)
        s = np.dot(x, x)
    return rq