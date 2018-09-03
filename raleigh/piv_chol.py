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

def piv_chol(A, k, eps, blk = 64, verb = 0):
    n = A.shape[0]
    buff = A[0,:].copy()
    ind = [i for i in range(n)]
    drop_case = 0
    dropped = 0
    last_check = -1
    if k > 0:
        U = sla.cholesky(A[:k,:k])
        A[:k,:k] = U.copy()
        A[:k, k : n] = sla.solve_triangular \
                       (conjugate(U), A[:k, k : n], lower = True)
        A[k : n, :k].fill(0.0)
        A[k : n, k : n] -= np.dot(conjugate(A[:k, k : n]), A[:k, k : n])
    l = k
    for i in range(k, n):
        s = np.diag(A[i : n, i : n]).copy()
        if i > l:
            t = nla.norm(A[l : i, i : n], axis = 0)
            s -= t*t
        j = i + np.argmax(s)
        if i != j:
            #print('swapping %d and %d' % (i, j))
            buff[:] = A[i,:]
            A[i,:] = A[j,:]
            A[j,:] = buff
            buff[:] = A[:,i]
            A[:,i] = A[:,j]
            A[:,j] = buff
            ind[i], ind[j] = ind[j], ind[i]
        if i > l:
            A[i, i : n] -= np.dot(conjugate(A[l : i, i]), A[l : i, i : n])
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
            lmin = estimate_lmin(A[: i + 1, : i + 1])
            lmax = estimate_lmax(A[: i + 1, : i + 1])
            if verb > 0:
                print('%e %e %e' % (A[i,i], lmin, lmax))
            if lmin/lmax <= eps:
                A[i : n, :].fill(0.0)
                drop_case = 2
                dropped = n - i
                break
        if i - l == blk - 1 and i < n - 1:
            j = i + 1
            A[j : n, j : n] -= np.dot(conjugate(A[l : j, j : n]), A[l : j, j : n])
            l += blk
    if last_check < n - 1 and drop_case != 2:
        i = last_check
        j = n - dropped - 1
        while i < j:
            m = i + (j - i + 1)//2
            lmin = estimate_lmin(A[: m + 1, : m + 1])
            lmax = estimate_lmax(A[: m + 1, : m + 1])
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
    return ind, dropped, last_piv

def piv_chol_old(A, k, eps, verb = 0):
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
