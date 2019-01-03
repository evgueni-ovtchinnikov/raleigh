# -*- coding: utf-8 -*-
"""Reads images from lfw folder or .npy file, erases background and saves in
.npy file.

Usage:
  preprocess_lfw [--help | -h | options] <datapath>

Arguments:
  datapath           lfw images folder

Options:
  -m, --how-many=<m>   number of images to process (<0: all) [default: -1]
  -o, --output=<file>  output file name [default: images.npy]
  -d, --double         double the number of images by adding mirror images
  -v, --view           view processed images

Created on Tue Aug 21 14:34:18 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

datapath = args['<datapath>']
m = int(args['--how-many'])
output = args['--output']
dble = args['--double']
view = args['--view']

import numpy
import numpy.linalg as nla
import os
import pylab
import scipy.ndimage as ndimage

def trim_mask(nx, ny):
    mask = numpy.zeros((ny, nx), dtype = numpy.uint8)
    x0 = nx/2
    y0 = ny/2
    ax = x0 - nx/5
    ay = y0 - ny/10
    for y in range(ny):
        for x in range(nx):
            if ((x - x0)/ax)**2 + ((y - y0)/ay)**2 > 1:
                mask[y, x] = 1
    return mask

print('loading images from %s...' % datapath)
image_dir = datapath
for subdir in os.listdir(image_dir):
    if subdir.endswith('.txt'):
        continue
    fulldir = image_dir + '/' + subdir
    for filename in os.listdir(fulldir):
        if not filename.endswith('.jpg'):
            continue
        fullname = fulldir + '/' + filename
        image = ndimage.imread(fullname, mode = 'L').astype(numpy.float32)
        break
    break

ny, nx = image.shape

print('counting images...')
ndir = 0
nimg = 0
for subdir in os.listdir(image_dir):
    if subdir.endswith('.txt'):
        continue
    fulldir = image_dir + '/' + subdir
    for filename in os.listdir(fulldir):
        if not filename.endswith('.jpg'):
            continue
        fullname = fulldir + '/' + filename
        nimg += 1
        if m > 0 and nimg >= m:
            break
    ndir += 1
    if m > 0 and nimg >= m:
        break

if dble:
    images = numpy.zeros((2*nimg, ny, nx), dtype = numpy.float32)
else:
    images = numpy.zeros((nimg, ny, nx), dtype = numpy.float32)
print('collecting %d images from %d folders...' % (nimg, ndir))
nimg = 0
for subdir in os.listdir(image_dir):
    if subdir.endswith('.txt'):
        continue
    fulldir = image_dir + '/' + subdir
    for filename in os.listdir(fulldir):
        if not filename.endswith('.jpg'):
            continue
        fullname = fulldir + '/' + filename
        image = ndimage.imread(fullname, mode = 'L')
        if dble:
            images[2*nimg, :, :] = image
        else:
            images[nimg, :, :] = image
        nimg += 1
        if m > 0 and nimg >= m:
            break
    if m > 0 and nimg >= m:
        break

vmax = numpy.amax(images)
vmin = numpy.amin(images)
print('pixel values range: %f to %f' % (vmin, vmax))
mask = trim_mask(nx, ny)
v = (vmax - vmin)/2
n = nx*ny
for i in range(nimg):
    image = images[2*i,:,:]
    image[mask > 0] = 0
    v = numpy.sum(image)/(n - numpy.sum(mask))
    image[mask > 0] = v
    images[2*i,:,:] = image
    if dble:
        images[2*i + 1, :, :] = image[:, ::-1]

print('saving %d images to %s...' % (images.shape[0], output))
numpy.save(output, images)

while view:
    i = int(input('image: '))
    if i < 0 or i >= images.shape[0]:
        break
    image = images[i,:,:]
#    print(numpy.mean(image))
    pylab.imshow(image, cmap = 'gray')
    pylab.show()

print('done')
