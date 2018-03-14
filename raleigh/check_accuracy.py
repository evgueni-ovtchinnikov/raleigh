import numpy
#import sys
#sys.path.append('..')
import raleigh.solver

def sort_eigenpairs(lmd, u, err_lmd, err_X):
    ind = numpy.argsort(lmd)
    w = u.new_vectors(u.nvec())
    lmd = lmd[ind]
    err_lmd = err_lmd[ind]
    err_X = err_X[ind]
    u.copy(w, ind)
    w.copy(u)
    return lmd, u, err_lmd, err_X

def check_eigenvectors_accuracy \
    (problem, opt, which = (-1,-1), \
                extra = (-1,-1), init = (None, None)):

    numpy.random.seed(1) # make results reproducible

    problem_type = problem.type()
    std = (problem_type == 's')

    u = problem.vector().new_vectors()
    v = problem.vector().new_vectors()

    solver = raleigh.solver.Solver(problem)

    # run first time
    solver.solve(u, opt, which, extra, init)
    print('after %d iterations, %d + %d eigenvalues converged:' \
          % (solver.iteration, solver.lcon, solver.rcon))
    lmdu = solver.eigenvalues
    lmdu, u, err_lmd, err_X = sort_eigenpairs\
        (lmdu, u, solver.errors_val, solver.errors_vec)
    print(lmdu)
    lconu = solver.lcon
    rconu = solver.rcon
    nconu = lconu + rconu

    # run second time
    solver.solve(v, opt, which, extra, init)
    print('after %d iterations, %d + %d eigenvalues converged:' \
          % (solver.iteration, solver.lcon, solver.rcon))
    lmdv = solver.eigenvalues
    lmdv, v, err_lmd, err_X = sort_eigenpairs\
        (lmdv, v, solver.errors_val, solver.errors_vec)
    print(lmdv)
    lconv = solver.lcon
    rconv = solver.rcon
    nconv = lconv + rconv

    # compare to get an idea about the actual eigenvector errors
    # since the error in each u and respective v are essentially random,
    # they are nearly orthogonal, hence the difference between u and v
    # is not less than the maximal of the two errors and not greater
    # than the latter multiplied by the square root of 2

    # compare eigenvectors on the left
    lcon = min(lconu, lconv)
    u.select(lcon)
    v.select(lcon)
    w = u.new_vectors(lcon)
    if std:
        q = v.dot(u)
        u.mult(q, w)
        w.add(v, -1.0)
        sl = w.dots(w)
    else:
        x = u.new_vectors(lcon)
        B = problem.B()
        B(v, w)
        q = w.dot(u)
        u.mult(q, w)
        w.add(v, -1.0)
        B(w, x)
        sl = x.dots(w)
    sl = numpy.sqrt(sl) # difference on the left
    tl = err_X[:lcon]

    # compare eigenvectors on the right
    rcon = min(rconu, rconv)
    u.select(rcon, nconu - rcon)
    v.select(rcon, nconv - rcon)
    w = u.new_vectors(rcon)
    if std:
        q = v.dot(u)
        u.mult(q, w)
        w.add(v, -1.0)
        sr = w.dots(w)
    else:
        x = u.new_vectors(rcon)
        B = problem.B()
        B(v, w)
        q = w.dot(u)
        u.mult(q, w)
        w.add(v, -1.0)
        B(w, x)
        sr = x.dots(w)
    sr = numpy.sqrt(sr) # difference on the right
    tr = err_X[nconv - rcon:]
    s = numpy.concatenate((sl, sr))
    t = numpy.concatenate((tl, tr))
    print('eigenvector errors:')
    print('  estimated    actual')
    for i in range(lcon + rcon):
        print('%e %e' % (t[i], s[i]))
    return s, t
