import math
import numpy
import scipy.linalg as sla
import sys
sys.path.append('..')

from raleigh.piv_chol import *

def op(x):
    n = x.shape[0]
    y = x.copy()
    for i in range(n):
        y[i] = (10**i) * x[i]
    return y/sla.norm(y)

n = 7
x = numpy.ones((n,n))/math.sqrt(n)
for j in range(n - 1):
    x[:, j + 1] = op(x[:, j])
A = numpy.dot(x.transpose(), x)
#print(x)
#print(A)

##A = numpy.zeros((n,n))
##for i in range(n):
##    A[i, i] = 2
##for i in range(n - 1):
##    A[i, i + 1] = -1
##    A[i + 1, i] = -1

U = A.copy()
ind, failed, last_piv = piv_chol(U, 0, 1e-15)
#print(U)
print('pivot order: ', ind)
print('failed rows: %d' % failed)
print('last_pivot: %e' % last_piv)

UTU = numpy.dot(U.transpose(), U)
#print(UTU)
#print(A[ind,:][:,ind])
print('factorization error: %e' % sla.norm(UTU - A[ind,:][:,ind]))

U = sla.cholesky(A)
#print(U)

