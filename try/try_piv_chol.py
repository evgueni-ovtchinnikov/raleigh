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

n = 6
x = numpy.ones((n,n)) #/math.sqrt(n)
for j in range(n - 1):
    x[:, j + 1] = op(x[:, j])
for j in range(n):
    x[:, j] /= sla.norm(x[:, j])
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
ind, failed, last_piv = piv_chol(U, 0, 1e-4)
k = n - failed
V = U[:k, :k]
print(V)
print('pivot order: ', ind)
print('failed rows: %d' % failed)
print('last_pivot squared: %e' % last_piv)

UTU = numpy.dot(U.transpose(), U)
#print(UTU)
#print(A[ind,:][:,ind])
print('full factorization error: %e' % sla.norm(UTU - A[ind,:][:,ind]))

y = x[:, ind[:n - failed]]
B = numpy.dot(y.transpose(), y)
VTV = numpy.dot(V.transpose(), V)
print('factorization error: %e' % sla.norm(VTV - B))

lmd, v = sla.eigh(B)
print(lmd)
print('condition number: %e' % (lmd[-1]/lmd[0]))
U = sla.cholesky(B)
#print(U)
#print(numpy.dtype(U[0,0]))
#estimate_lmin(U)
#UTU = numpy.dot(U.transpose(), U)
#lmd, v = sla.eigh(UTU)
#print(lmd)
#print('condition number: %e' % (lmd[-1]/lmd[0]))

y, r = sla.qr(y, mode = 'economic')
z = x - numpy.dot(y, numpy.dot(y.transpose(), x))
#print(x.shape)
#print(y.shape)
#print(z.shape)
for i in range(n):
    print(sla.norm(z[:,i])/sla.norm(x[:,i]))

try:
    U = sla.cholesky(A)
    #print(U)
except:
    print('factorization failed')
