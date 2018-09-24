import numpy
from scipy.sparse.linalg import LinearOperator, lobpcg
import time

def random_singular_values(k, sigma, dt):
    s = numpy.random.rand(k).astype(dt)
    s = numpy.sort(s)
    t = numpy.ones(k)*s[0]
    return sigma(k*(s - t))

def random_singular_vectors(m, n, k, dt):
    u = numpy.random.randn(m, k).astype(dt)
    v = numpy.random.randn(n, k).astype(dt)
    u, r = numpy.linalg.qr(u)
    v, r = numpy.linalg.qr(v)
    return u, v

def random_matrix_for_svd(m, n, k, sigma, dt):
    u, v = random_singular_vectors(m, n, k, dt)
    s = random_singular_values(k, sigma, dt).astype(dt)
    a = numpy.dot(u*s, v.transpose())
    return s, u, v, a

numpy.random.seed(1)

dtype = numpy.float32 #64

m = 15000
n = 40000
l = 1000
alpha = 0.01
f_sigma = lambda t: 2**(-alpha*t).astype(dtype)
print('\n--- generating the matrix...')
sigma0, u0, v0, A = random_matrix_for_svd(m, n, l, f_sigma, dtype)

k = 20
print('\n %d largest singular values of A:' % k)
print(sigma0[:k])

def mv(x):
    return -numpy.dot(A, numpy.dot(A.T, x))

opA = LinearOperator \
    (dtype = A.dtype, shape = (m, m), matvec = mv, matmat = mv)
X = numpy.random.randn(m, k).astype(A.dtype)
# largest is ignored!
print('\n--- solving...')
start = time.time()
lmd, x = lobpcg(opA, X, largest = False, maxiter = 100, tol = 1e-3, verbosityLevel = 0)
#lmd, x = lobpcg(opA, X, largest = True, maxiter = 100, tol = 1e-6)
#lmd, x = lobpcg(opA, X, largest = False, maxiter = 100, tol = 1e-4, verbosityLevel = 0)
#lmd, x = lobpcg(opA, X, largest = True, maxiter = 100, tol = 1e-4, verbosityLevel = 0)
stop = time.time()
el_time = stop - start
print('\n %d largest singular values from lobpcg:' % k)
#print(lmd)
print(numpy.sqrt(abs(lmd)))
print('elapsed time %.1e' % el_time)
