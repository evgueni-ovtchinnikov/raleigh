import numpy
import sys

import try_vectors

sys.path.append('..')

from raleigh.ndarray_vectors import NDArrayVectors

n = 9
m = 4
#array = new_orthogonal_vectors(n, m)
#print(array)
v = NDArrayVectors(numpy.zeros((0, n), order = 'C'))
try_vectors.test(v, m)

#n = 50000
#m = 1000
#v = NDArrayVectors(numpy.zeros((1, n), order = 'C'))
#try_vectors.ptest(v, m)
