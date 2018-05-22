import numpy
import scipy.linalg as sla
import raleigh.solver

def sort_eigenpairs_and_errors(lmd, u, err_lmd, err_X, res, cnv_stat):
    ind = numpy.argsort(lmd)
    w = u.new_vectors(u.nvec())
    lmd = lmd[ind]
    u.copy(w, ind)
    w.copy(u)
    err_lmd.reorder(ind)
    err_X.reorder(ind)
    res = res[ind]
    cnv_stat = cnv_stat[ind]
    return lmd, u, err_lmd, err_X, res, cnv_stat

def RayleighRitz(A, u):
    m = u.nvec()
    v = u.new_vectors(m)
    w = u.new_vectors(m)
    p = u.dot(u)
    q = p - numpy.eye(p.shape[0])
    print(numpy.linalg.norm(q))
    A(u, v)
    q = v.dot(u)
#    lmd = numpy.diag(q)
    lmd, Q = sla.eigh(q, p)
    u.multiply(Q, w)
    w.copy(u)
    v.multiply(Q, w)
    w.copy(v)
    q = v.dot(u)
    u.multiply(q, w)
    w.add(v, -1.0)
    t = w.dots(w)
    t = numpy.sqrt(t)
    v.add(u, -lmd)
    s = v.dots(v)
    s = numpy.sqrt(s)
    return s, t

def check_eigenvectors_accuracy \
    (problem, opt, which = (-1,-1), \
                extra = (-1,-1), init = (None, None), v_ex = None):

    numpy.random.seed(1) # make results reproducible

    problem_type = problem.type()
    std = (problem_type == 's')

    solver = raleigh.solver.Solver(problem)

    # run first time
    u = problem.vector().new_vectors()
    solver.solve(u, opt, which, extra, init)
    print('%d + %d eigenvalues computed in %d iterations:' \
          % (solver.lcon, solver.rcon, solver.iteration))
    lmdu = solver.eigenvalues
#    print(lmdu)
    lmdu, u, err_lmdu, err_u, res_u, cnv_u = sort_eigenpairs_and_errors\
        (lmdu, u, solver.eigenvalue_errors, solver.eigenvector_errors, \
        solver.residual_norms, solver.convergence_status)
    print(lmdu)
    lconu = solver.lcon
    rconu = solver.rcon
    nconu = lconu + rconu
    if nconu < 1:
        print('no eigenpairs converged')
        return

    if v_ex is None:
        # run second time
        v = problem.vector().new_vectors()
        solver.solve(v, opt, which, extra, init)
        print('%d + %d eigenvalues computed in %d iterations:' \
              % (solver.lcon, solver.rcon, solver.iteration))
        lmdv = solver.eigenvalues
        lmdv, v, err_lmdv, err_v, res_v, cnv_v = sort_eigenpairs_and_errors\
            (lmdv, v, solver.eigenvalue_errors, solver.eigenvector_errors, \
            solver.residual_norms, solver.convergence_status)
        print(lmdv)
        lconv = solver.lcon
        rconv = solver.rcon
        nconv = lconv + rconv
        if nconv < 1:
            print('no eigenpairs converged')
            return
    else:
        v = v_ex
        n_ex = v_ex.nvec()
        lconv = n_ex
        rconv = n_ex
        nconv = n_ex
        
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
    if lcon > 0:
        u.select(lcon)
        v.select(lcon)
        w = u.new_vectors(lcon)
        if std:
            q = v.dot(u)
            u.multiply(q, w)
            w.add(v, -1.0)
            sl = w.dots(w)
        else:
            x = u.new_vectors(lcon)
            B = problem.B()
            B(v, w)
            q = w.dot(u)
            u.multiply(q, w)
            w.add(v, -1.0)
            B(w, x)
            sl = x.dots(w)
        sl = numpy.sqrt(sl) # difference on the left

    # compare eigenvectors on the right
    rcon = min(rconu, rconv)
    u.select(rcon, nconu - rcon)
    v.select(rcon, nconv - rcon)
    w = u.new_vectors(rcon)

#    A = problem.A()
##    u.copy(w)
#    s, t = RayleighRitz(A, u)
##    r = w.dots(u)
##    w.add(u, -r)
##    print(numpy.sqrt(w.dots(w)))
##    for i in range(rcon):
##        print('%e  %.1e  %.1e  %.1e' % \
##        (lmdu[nconu - rcon + i], res_u[nconu - rcon + i], s[i], t[i]))
#    s, t = RayleighRitz(A, v)
##    for i in range(rcon):
##        print('%.1e  %.1e' % (s[i], t[i]))
##    for i in range(rcon):
##        print('%e  %.1e  %.1e  %.1e' % \
##        (lmdv[nconv - rcon + i], res_v[nconv - rcon + i], s[i], t[i]))

    if rcon > 0:
        if std:
            q = v.dot(u)
            u.multiply(q, w)
            w.add(v, -1.0)
            sr = w.dots(w)
        else:
            x = u.new_vectors(rcon)
            B = problem.B()
            B(v, w)
            q = w.dot(u)
            u.multiply(q, w)
            w.add(v, -1.0)
            B(w, x)
            sr = x.dots(w)
        sr = numpy.sqrt(sr) # difference on the right

    print('eigenvector errors:')
    msg = ' eigenvalue   estimate (kinematic/residual)   actual' + \
        '       status'
    print(msg)
    for i in range(lcon):
        lmdu_i = lmdu[i]
        err_ui = err_u[i]
        err_uik = abs(err_ui[0])
        err_uir = abs(err_ui[1])
        if v_ex is None:
            err_vi = err_v[i]
            err_vik = abs(err_vi[0])
            err_vir = abs(err_vi[1])
        else:
            err_vik = 0
            err_vir = 0
        err_ik = max(err_uik, err_vik)
        err_ir = max(err_uir, err_vir)
        if cnv_u[i] > 0:
            st = 'converged at'
        elif cnv_u[i] < 0:
            st = 'stagnated at'
        else:
            st = ' '
        print('%e        %.1e / %.1e        %.1e    %s %d' % \
            (lmdu_i, err_ik, err_ir, sr[i], st, abs(cnv_u[i]) - 1))
    for j in range(rcon):
        i = nconu - rcon + j
        lmdu_i = lmdu[i]
        err_ui = err_u[i]
        err_uik = abs(err_ui[0])
        err_uir = abs(err_ui[1])
        if v_ex is None:
            err_vi = err_v[nconv - rcon + j]
            err_vik = abs(err_vi[0])
            err_vir = abs(err_vi[1])
        else:
            err_vik = 0
            err_vir = 0
        err_ik = max(err_uik, err_vik)
        err_ir = max(err_uir, err_vir)
        if cnv_u[i] > 0:
            st = 'converged at'
        elif cnv_u[i] < 0:
            st = 'stagnated at'
        else:
            st = ' '
        print('%e        %.1e / %.1e        %.1e    %s %d' % \
            (lmdu_i, err_ik, err_ir, sr[i], st, abs(cnv_u[i]) - 1))
