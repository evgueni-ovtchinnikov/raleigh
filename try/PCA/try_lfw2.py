import math
import numpy
import numpy.linalg as nla
import os
import pylab
import scipy.sparse.linalg as sla
import scipy.ndimage as ndimage

dir = 'C:/Users/wps46139/Documents/Data/PCA/'

nsv = 234
sigma = numpy.load(dir + 'sigma_%d.npy' % nsv)
u = numpy.load(dir + 'u_%d.npy' % nsv)
v = numpy.load(dir + 'v_%d.npy' % nsv)
print(sigma.shape)
print(u.shape)
print(v.shape)

nsv, ny, nx = u.shape
m = v.shape[1]
n = nx*ny

##umax = 1/math.sqrt(n) # imshow range
##
##while True:
##    i = int(input('singular pair number (negative to stop): '))
##    if i < 0 or i >= nsv:
##        break
##    pylab.figure(i)
##    pylab.title('PC %d, left singular vector' % i)
####    s = sigma[i]/sigma[0]
##    pylab.imshow(u[i,:,:], cmap = 'gray')
####    pylab.imshow(u[i,:,:], vmin = -umax, vmax = umax, cmap = 'gray')
##    pylab.show()

u = numpy.reshape(u, (nsv, n))

a = numpy.load(dir + 'lfw2.npy')
a0 = numpy.load(dir + 'lfw2_skip.npy')
m0 = a0.shape[0]
print(m0)
#a = numpy.load(dir + 'lfw2.npy')
#print(a.shape)
#b = numpy.dot(v.T*sigma, u)
#print(b.shape)
#b = numpy.reshape(b, (m, ny, nx))

a1 = numpy.reshape(a, (m, n))

while True:
    i = int(input('image: '))
    if i < 0 or i >= m0:
        break
    image = a0[i,:,:]
    pylab.figure()
    pylab.imshow(image, cmap = 'gray')
##    pylab.figure()
##    pylab.imshow(b[i,:,:], cmap = 'gray')
##    pylab.show()
    d = numpy.reshape(image, (n,))
    w = numpy.dot(u, d)
##    print(w.shape)
    h = numpy.dot(w.T, v)
#    print(h.shape)
    j = numpy.argmax(abs(h))
##    ind = numpy.argsort(h)
##    for k in range(5):
##        j = ind[-1 - k]
    print(j, h[j])
##    p = numpy.dot(a1, d)/nla.norm(a1, axis = 1)
##    j = numpy.argmax(abs(p))
##    print(j, p[j])
    image_found = a[j,:,:]
    pylab.figure()
    pylab.imshow(image_found, cmap = 'gray')
    p = sigma*v[:,j]
##    print(p.shape)
    img = numpy.reshape(numpy.dot(u.T, p), (ny, nx))
    pylab.figure()
    pylab.imshow(img, cmap = 'gray')
    pylab.show()


