'''Test partial SVD.

Tests partial SVD of the set of 2D images <data>.

SVD data is read from files sigma_<s>.npy, u_<s>.npy and v_<s>.npy.

Usage:
    test_psvd [--help | -h | options] <data> <s>

Arguments:
    data  data set file
    s     partial svd data suffix

Options:
  -p <path>, --path=<path>   path to the data directory
                             [default: C:/Users/wps46139/Documents/Data/PCA]
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

nsv = args['<s>']
dir = args['--path']
data = args['<data>']

import numpy
import numpy.linalg as nla
import os
import pylab

a = numpy.load(dir + '/' + data + '.npy')
ma, nya, nxa = a.shape

sigma = numpy.load(dir + '/sigma_%s.npy' % nsv)
u = numpy.load(dir + '/u_%s.npy' % nsv)
v = numpy.load(dir + '/v_%s.npy' % nsv)

nsv, ny, nx = u.shape
m = v.shape[1]
n = nx*ny

if m != ma or nx != nxa or ny != nya:
    raise ValueError('data sizes (%d, %d, %d) and (%d, %d, %d) do not match' % \
          (m, ny, nx, ma, nya, nxa))

u = numpy.reshape(u, (nsv, n))

ns = 100

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= m:
        break
    image = a[i,:,:]
    pylab.figure()
    pylab.title('Photo to be searched for')
    pylab.imshow(image, cmap = 'gray')
    ai = numpy.reshape(image, (n,))
    ai = ai/nla.norm(ai)
    w = numpy.dot(u, ai)
    h = numpy.dot(w.T/sigma, v)
##    j = numpy.argmax(abs(h))
    ind = numpy.argsort(h)
    phi_max = 0
    for k in range(ns):
        js = ind[-1 - k]
        ajs = a[js,:,:]
        ajs = numpy.reshape(ajs, (n,))
        ajs = ajs/nla.norm(ajs)
        phi = numpy.dot(ai, ajs)
##        print(js, h[js], phi)
        if phi > phi_max:
            phi_max = phi
            j = js
            if phi > 0.99999:
                print('found at step %d' % k)
                break
    aj = a[j,:,:]
    pylab.figure()
    pylab.title('Photo found')
    pylab.imshow(aj, cmap = 'gray')
    psvd_img = numpy.reshape(numpy.dot(u.T, sigma*v[:,j]), (ny, nx))
    pylab.figure()
    pylab.title('SVD approximation of the photo')
    pylab.imshow(psvd_img, cmap = 'gray')
    pylab.show()
