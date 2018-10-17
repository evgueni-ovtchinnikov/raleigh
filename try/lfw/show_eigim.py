'''Shows eigenimages.

Usage:
    show_eigim [--help | -h | options] <file> <tiles>

Arguments:
    file       eigenimages file (must end with .npy)

Created on Wed Aug 22 11:13:17 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

filename = args['<file>']
nt = int(args['<tiles>'])

import numpy
import pylab

print('loading eigenimages from %s...' % filename)
u = numpy.load(filename)

nu, ny, nx = u.shape
print('%d eigenimages of size %dx%d loaded' % (nu, ny, nx))

images = numpy.ndarray((nt, ny, nt, nx), dtype = u.dtype)

while True:
    i = int(input('first eigenimage number (negative to exit): '))
    if i < 0 or i >= nu:
        break
    j = i
    for ty in range(nt):
        for tx in range(nt):
            if j < nu:
                images[ty, :, tx, :] = u[j,:,:]
            else:
                images[ty, :, tx, :] = 0
            j += 1
    image = numpy.reshape(images, (nt*ny, nt*nx))
    pylab.figure()
    pylab.title('eigenimages %d to %d' % (i, j - 1))
    pylab.imshow(image, cmap = 'gray')
    pylab.show()
