'''
Test the use of abstract vectors

'''

import time

def test(x, n):
    print(x.data_type())
    v = x.new_orthogonal_vectors(n)
    print(type(v))
    print(v.data().flags)
    print('v:')
    print(v.data())
    q = v.dot(v)
    print('v\'*v:')
    print(q)
    #return
    w = v.new_vectors(n)
    v.select(2, 1)
    s = v.selected()
    print(s)
    v.copy(w)
    print(v.data().flags)
    print(w.data())
    print(v.data())
    q = w.dot(v)
    print('v\'*w:')
    print(q)
    print('w\'*w:')
    print(w.selected())
    s = w.dots(w)
    print(s)
    w.scale(s)
    print('w/s:')
    print(w.data())
    v.mult(q, w)
    print('v*q:')
    print(w.data())
    v.select(n, 0)
    print('v:')
    print(v.data())
    v.add(w, s)
    print('v:')
    print(v.data())

def ptest(x, m):
    v = x.new_orthogonal_vectors(m)
    print(v.data().shape)
    w = x.new_vectors(m)
    print(w.data().shape)
    v.copy(w)
    #v.select(1, m - 1)
    w.select(1, m - 1)
    print('v flags:')
    print(v.data().flags)
    print('w flags:')
    print(w.data().flags)
    q = w.dot(v)
    start = time.time()
    v.mult(q, w)
    stop = time.time()
    elapsed_time = stop - start
    print('elapsed time: %e' % elapsed_time)
