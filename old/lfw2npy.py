import numpy
import os
import scipy.ndimage as ndimage

data_root = 'C:/Users/wps46139/Documents/Data/PCA/'
lfw2_dir = data_root + 'lfw2/'

ex_prefix = 'George_W_Bush_'
exclude = []
for suffix in ['0038', '0051', '0078']:
    exclude += [ex_prefix + suffix + '.jpg']
#print(exclude)

print('counting...')
ndir = 0
nkeep = 0
nskip = 0
for subdir in os.listdir(lfw2_dir):
    fulldir = lfw2_dir + subdir
    for filename in os.listdir(fulldir):
        if filename in exclude:
            print('skipping %s' % filename)
            nskip += 1
        elif filename.endswith('.jpg'):
            nkeep += 1
    ndir += 1

print('collecting %d files from %d folders..' % (nkeep, ndir))
a = numpy.zeros((nkeep, 250, 250), dtype = numpy.float32)
b = numpy.zeros((nskip, 250, 250), dtype = numpy.float32)
nkeep = 0
nskip = 0
for subdir in os.listdir(lfw2_dir):
    fulldir = lfw2_dir + subdir
    print(fulldir)
    for filename in os.listdir(fulldir):
        if filename in exclude:
            print('skipping %s' % filename)
            b[nskip,:,:] = ndimage.imread(fulldir + '/' + filename)
            nskip += 1
        elif filename.endswith('.jpg'):
            a[nkeep,:,:] = ndimage.imread(fulldir + '/' + filename)
            nkeep += 1

print('saving...')
numpy.save('lfw2_keep.npy', a)
numpy.save('lfw2_skip.npy', b)
