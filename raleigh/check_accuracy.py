import numpy
#import sys
#sys.path.append('..')
import raleigh.solver

def sort_eigenpairs_and_errors(lmd, u, err_lmd, err_X, res):
    ind = numpy.argsort(lmd)
    w = u.new_vectors(u.nvec())
    lmd = lmd[ind]
    u.copy(w, ind)
    w.copy(u)
    err_lmd.reorder(ind)
    err_X.reorder(ind)
    res = res[ind]
    return lmd, u, err_lmd, err_X, res

def check_eigenvectors_accuracy \
    (problem, opt, which = (-1,-1), \
                extra = (-1,-1), init = (None, None)):

    numpy.random.seed(1) # make results reproducible

    problem_type = problem.type()
    std = (problem_type == 's')

    solver = raleigh.solver.Solver(problem)

    # run first time
    u = problem.vector().new_vectors()
    solver.solve(u, opt, which, extra, init)
    print('after %d iterations, %d + %d eigenvalues converged:' \
          % (solver.iteration, solver.lcon, solver.rcon))
    lmdu = solver.eigenvalues
    lmdu, u, err_lmdu, err_u, res_u = sort_eigenpairs_and_errors\
        (lmdu, u, solver.eigenvalue_errors, solver.eigenvector_errors, \
        solver.residual_norms)
    print(lmdu)
    lconu = solver.lcon
    rconu = solver.rcon
    nconu = lconu + rconu
    if nconu < 1:
        print('no eigenpairs converged')
        return

    # run second time
    v = problem.vector().new_vectors()
    solver.solve(v, opt, which, extra, init)
    print('after %d iterations, %d + %d eigenvalues converged:' \
          % (solver.iteration, solver.lcon, solver.rcon))
    lmdv = solver.eigenvalues
    lmdv, v, err_lmdv, err_v, res_v = sort_eigenpairs_and_errors\
        (lmdv, v, solver.eigenvalue_errors, solver.eigenvector_errors, \
        solver.residual_norms)
    print(lmdv)
    lconv = solver.lcon
    rconv = solver.rcon
    nconv = lconv + rconv
    if nconv < 1:
        print('no eigenpairs converged')
        return
        
#    for i in range(nconu):
#        print('eigenvalue: %e, residual norm: %.1e' % (lmdu[i], res_u[i]))
#    for i in range(nconv):
#        print('eigenvalue: %e, residual norm: %.1e' % (lmdv[i], res_v[i]))

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

    print('eigenvector errors:')
    msg = '   estimated (first | second, kinematic/residual)' + \
    '                actual'
    print(msg)
    for i in range(lcon):
        err_ui = err_u[i]
        err_vi = err_v[i]
        print('%e / %e | %e / %e    %e' % \
        (err_ui[0], err_ui[1], err_vi[0], err_vi[1], sl[i]))
    for i in range(rcon):
        err_ui = err_u[nconu - rcon + i]
        err_vi = err_v[nconv - rcon + i]
        print('%e / %e | %e / %e    %e' % \
        (err_ui[0], err_ui[1], err_vi[0], err_vi[1], sr[i]))
