'''Self-adjoint operators library
'''

class Diagonal:
    def __init__(self, d):
        self.d = d
        self.n = d.shape[0]
    def apply(self, x, y):
        y.data()[:,:] = self.d * x.data()
