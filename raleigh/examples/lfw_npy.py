# -*- coding: utf-8 -*-
"""Reads images from lfw folder or .npy file, processes and saves in .npy file.

Usage:
  lfw_npy [--help | -h | options] <datapath>

Arguments:
  datapath           lfw images folder or .npy file

Options:
  -o, --output=<file>  output file name [default: images.npy]
  -n, --normalize      l2-normalize the images
  -s, --shift          shift to zero-average
  -f, --focus          zero outside face area
  -v, --view           view processed images

Created on Tue Aug 21 14:34:18 2018

@author: Evgueni Ovtchinnikov, UKRI-STFC
"""

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

normalize = args['--normalize']
shift = args['--shift']
focus = args['--focus']
view = args['--view']
output = args['--output']
datapath = args['<datapath>']

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

if datapath.endswith('.npy'):

    print('loading images from %s...' % datapath)
    images = numpy.load(datapath)
    nimg, ny, nx = images.shape

else:

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
        ndir += 1
    
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
            images[nimg,:,:] = ndimage.imread(fullname, mode = 'L')
            nimg += 1

if normalize or shift or focus:
    print('preprocessing images...')
    n = nx*ny
    mask = trim_mask(nx, ny)
    for i in range(nimg):
        image = images[i,:,:]
        if shift:
            s = numpy.sum(image)/n
            image -= s
        if focus:
            image[mask > 0] = 0
        if normalize:
            s = nla.norm(numpy.reshape(image, (n, 1)))
            image = image/s
        images[i,:,:] = image

print('saving to %s...' % output)
numpy.save(output, images)

while view:
    i = int(input('image: '))
    if i < 0 or i >= nimg:
        break
    image = images[i,:,:]
    pylab.imshow(image, cmap = 'gray')
    pylab.show()

print('done')