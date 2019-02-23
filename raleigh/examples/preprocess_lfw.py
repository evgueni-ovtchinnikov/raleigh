# -*- coding: utf-8 -*-
"""Reads images from lfw folder or .npy file, optionally processes them and 
   saves in .npy file.

Usage:
  preprocess_lfw [--help | -h | options] <datapath>

Arguments:
  datapath           lfw images folder

Options:
  -m, --how-many=<m>   number of images to process (<0: all) [default: -1]
  -o, --output=<file>  output file name [default: images.npy]
  -s, --asymm=<s>      select images that differ from their mirror images
                       by less than either (i) <s> times maximal difference,
                       if s > 0 or (ii) mean difference and save them
                       to photos.npy [default: 0.5]
  -d, --double         double the number of images by adding mirror images
  -f, --face-area      set pixels outside face area to average value
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
asymm = float(args['--asymm'])
dble = args['--double']
face = args['--face-area']
view = args['--view']

import numpy
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

if dble:
    ni = 2*nimg
else:
    ni = nimg
a = numpy.zeros(ni)
ia = numpy.zeros(ni, dtype = numpy.int32)
mask = trim_mask(nx, ny)
n = nx*ny
for i in range(nimg):
    if dble:
        j = 2*i
    else:
        j = i
    image = images[j,:,:]
    if face:
        image[mask > 0] = 0
        v = numpy.sum(image)/(n - numpy.sum(mask))
        image[mask > 0] = v
        images[j,:,:] = image
    a[j] = numpy.linalg.norm(image - image[:, ::-1])
    ia[j] = j
    if dble:
        images[j + 1, :, :] = image[:, ::-1]
        a[j + 1] = a[j]
        ia[j + 1] = j + 1
ind = numpy.argsort(a)
pylab.figure()
pylab.plot(numpy.arange(1, ni + 1, 1), a[ind])
pylab.show()
max_asymm = numpy.amax(a)
mean_asymm = numpy.mean(a)
print('asymmetry: max %f, mean %f' % (max_asymm, mean_asymm))
if asymm > 0:
    th = max_asymm*asymm
else:
    th = mean_asymm
k = sum(a < th)
#photos = images[ind[:k],:,:]
iind = numpy.sort(ia[ind[:k]])
photos = images[iind,:,:]
print(photos.shape)
while True:
    i = int(input('image: '))
    if i < 0 or i >= k:
        break
    image = photos[i,:,:]
    pylab.imshow(image, cmap = 'gray')
    pylab.show()
print('saving %d photos to %s...' % (k, 'photos.npy'))
numpy.save('photos.npy', photos)

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
