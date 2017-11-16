'''
Test the use of abstract vectors

'''

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
    v.select(1, 2)
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
    s = w.dots(w)
    print(s)
    w.scale(s)
    print('w*s:')
    print(w.data())
    v.mult(q, w)
    print('v*q:')
    print(w.data())
    v.select(0, n)
    print('v:')
    print(v.data())
    v.add(w, s)
    print('v:')
    print(v.data())
