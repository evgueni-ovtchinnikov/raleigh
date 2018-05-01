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

    q = u.dot(u)
    q = q - numpy.eye(q.shape[0])
    print(numpy.linalg.norm(q))
    q = v.dot(v)
    q = q - numpy.eye(q.shape[0])
    print(numpy.linalg.norm(q))
    
    z = u.new_vectors(rcon)
    A = problem.A()
    A(u, w)
    q = w.dot(u)
    p = q - numpy.diag(lmdu[nconu - rcon :])
    print(numpy.linalg.norm(p))
    u.mult(q, z)
    z.add(w, -1.0)
    t = z.dots(z)
    t = numpy.sqrt(t)
    w.add(u, -lmdu[nconu - rcon :])
    s = w.dots(w)
    s = numpy.sqrt(s)
    for i in range(rcon):
        print('%e  %.1e  %.1e  %.1e' % \
        (lmdu[nconu - rcon + i], res_u[nconu - rcon + i], s[i], t[i]))

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
    msg = ' eigenvalue  estimated (kinematic/residual)' + \
          '   actual'
    print(msg)
#    print('     first pass             second pass')
    for i in range(lcon):
        lmdu_i = lmdu[i]
        err_ui = err_u[i]
        err_uik = abs(err_ui[0])
        err_uir = abs(err_ui[1])
        err_vi = err_v[i]
        err_vik = abs(err_vi[0])
        err_vir = abs(err_vi[1])
        err_ik = max(err_uik, err_vik)
        err_ir = max(err_uir, err_vir)
        print('%e        %.1e / %.1e        %.1e' % \
        (lmdu_i, err_ik, err_ir, sr[i]))
#        print('  %.1e / %.1e      %.1e / %.1e         %.1e' % \
#        (abs(err_ui[0]), abs(err_ui[1]), abs(err_vi[0]), abs(err_vi[1]), sl[i]))
    for i in range(rcon):
        lmdu_i = lmdu[nconu - rcon + i]
        err_ui = err_u[nconu - rcon + i]
        err_uik = abs(err_ui[0])
        err_uir = abs(err_ui[1])
#        res_ui = res_u[nconu - rcon + i]
#        lmdv_i = lmdv[nconv - rcon + i]
        err_vi = err_v[nconv - rcon + i]
        err_vik = abs(err_vi[0])
        err_vir = abs(err_vi[1])
#        res_vi = res_v[nconv - rcon + i]
        err_ik = max(err_uik, err_vik)
        err_ir = max(err_uir, err_vir)
        print('%e        %.1e / %.1e        %.1e' % \
        (lmdu_i, err_ik, err_ir, sr[i]))
#        print('%e %.1e  %.1e / %.1e      %e %.1e  %.1e / %.1e     %.1e' % \
#        (lmdu_i, res_ui, err_uik, err_uir, lmdv_i, res_vi, err_vik, err_vir, sr[i]))
