'''Demostrates the assembly of PCA approximations from PC coordinates of images
   and the computation of PC coordinates of a given image.

Tha shape of images array loaded from an .npy file is (ni, ny, nx), where ni
is the number of images and (nx, ny) are image dimensions.

PCA approximation to images[i,:,:] is computed as the sum of u[j,:,:]*v[j, i]
where u[j,:,:] are PCs (= eigenimages = eigenfaces) and v[j,i] are PC
coordinates (= eigenfaces weights) of images[i,:,:].

The shape of u is (nc, ny, nx), where nc is the number of PCs.

The shape of v is (nc, ni).

PCs u are loaded from the file <prefix>eigim.npy and PC coordinates v
from the file <prefix>coord.npy.

Usage:
    assemble [--help | -h | options] <images> <prefix>

Arguments:
    images   images file
    prefix   PCs and PC coordinates prefix

Options:
  -p <path>, --path=<path>  path to input files [default: .]

@author: Evgueni Ovtchinnikov, UKRI-STFC
'''

import numpy
import pylab

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

file = args['<images>']
pref = args['<prefix>']
path = args['--path']

print('loading images from %s...' % (path + '/' + file))
images = numpy.load(file)
ni, ny, nx = images.shape

# link images numbers to names
names = []
index = numpy.ndarray((ni,), dtype = numpy.int16)
off = []
count = 0
with open('lfw_names.txt') as fp:
    line = fp.readline()
    while line:
        k = line.find(' ')
        names += [line[:k].replace('_', ' ')]
        new_off = int(line[k:])
        off += [new_off]
        count += 1
        line = fp.readline()
off += [ni]
i = 0
for c in range(count):
    for j in range(off[c], off[c + 1]):
        index[j] = i
    i += 1

filename = path + '/%seigim.npy' % pref
print('loading eigenimages from %s...' % filename)
u = numpy.load(filename)

filename = path + '/%scoord.npy' % pref
print('loading images coordinates in eigenimages basis from %s...' % filename)
v = numpy.load(filename)

nc, ny, nx = u.shape
ni = v.shape[1]

n = nx*ny
u = numpy.reshape(u, (nc, n))

while True:
    i = int(input('image number (negative to exit): '))
    if i < 0 or i >= ni:
        break

    pylab.figure()
    pylab.title('image %d (%s)' % (i, names[index[i]]))
    image = images[i,:,:]
    pylab.imshow(image, cmap = 'gray')

    # this is how the PC coordinates of image are computed:
    image = numpy.reshape(image, (n,))
    w = numpy.dot(u, image)
    # since v[:,i] holds the coordinates of images[i,:,:], the computed
    # coordinates w must coincide with v[:,i] modulo the PCA error
    print('coordinate error: %.1e' % numpy.linalg.norm(w - v[:,i]))

    # this is how the PCA approximation of image i is assembled from its
    # PCs coordinates v[:,i] and PCs u:
    PCA_image = numpy.dot(u.T, v[:,i])
    PCA_image = numpy.reshape(PCA_image, (ny, nx))
    pylab.figure()
    pylab.title('PCA approximation of the image %d' % i)
    pylab.imshow(PCA_image, cmap = 'gray')
    pylab.show()

