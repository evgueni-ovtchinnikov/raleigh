import math
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

def piv_chol(A, k, eps):
    n = A.shape[0]
    ind = [i for i in range(n - k)]
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
                ind[i - k], ind[j - k] = ind[j - k], ind[i - k]
            A[i, i : n] -= np.dot(A[:i, i].transpose(), A[:i, i : n])
        #print('pivot: %e' % A[i,i])
        if A[i, i] <= eps:
            A[i : n, :].fill(0.0)
            return ind, n - i, A[i, i]
        A[i, i] = math.sqrt(A[i, i])
        A[i, i + 1 : n] /= A[i, i]
        A[i + 1 : n, i].fill(0.0)
    return ind, 0, A[n - 1, n - 1]
