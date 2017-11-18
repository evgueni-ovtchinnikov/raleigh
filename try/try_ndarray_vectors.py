import numpy
import sys

import try_vectors

sys.path.append('..')

from raleigh.ndarray_vectors_opt import NDArrayVectors

##def orthogonal_vectors(n, m):
##    w = numpy.ndarray((n, m))
##    step = 1
##    for j in range(m):
##        for i in range(n):
##            k = i//step + 1
##            if k%2 == 1:
##                w[i,j] = 1
##            else:
##                w[i,j] = -1
##        step *= 2
##    return w

#def orthogonal_vectors(n, m):
#    if n < m:
#        print('Warning: number of vectors too large, reducing')
#        m = n
#    a = numpy.ones((1,1))
#    i = 1
#    while 2*i < m:
#        b = numpy.concatenate((a, a))
#        c = numpy.concatenate((a, -a))
#        a = numpy.concatenate((b, c), axis = 1)
#        i *= 2
#    if i < m:
#        a = numpy.concatenate((a, numpy.zeros((i, m - i))), axis = 1)
#    j = i
#    while 2*i < n:
#        a = numpy.concatenate((a, a))
#        i *= 2
#    if i < n:
#        a = numpy.concatenate((a, numpy.zeros((n - i, m))))
#    while i < n and j < m:
#        a[i,j] = 1.0
#        i += 1
#        j += 1
#    return a
#
#def new_orthogonal_vectors(n, m):
#    if n < m:
#        print('Warning: number of vectors too large, reducing')
#        m = n
#    a = numpy.zeros((n, m))
#    a[0,0] = 1.0
#    i = int(1)
#    while 2*i <= m:
#        a[i : 2*i, :i] = a[:i, :i]
#        a[:i, i : 2*i] = a[:i, :i]
#        a[i : 2*i, i : 2*i] = -a[:i, :i]
#        i *= 2
#    k = i
#    j = 2*i
#    if j > n:
#        for i in range(k, m):
#            a[i, i] = 1
#        return a
#    while j <= n:
#        a[i : j, :i] = a[:i, :i]
#        i, j = j, 2*j
#    j = i/2
#    a[ : j, k : m] = a[ : j, :(m - k)]
#    a[j : i, k : m] = -a[j : i, :(m - k)]
#    return a

#n = 9
#m = 4
##array = new_orthogonal_vectors(n, m)
##print(array)
#v = NDArrayVectors(numpy.zeros((n, 1), order = 'C'))
#try_vectors.test(v, m)

n = 50000
m = 1000
v = NDArrayVectors(numpy.zeros((1, n), order = 'C'))
try_vectors.ptest(v, m)
