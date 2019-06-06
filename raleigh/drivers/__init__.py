'''Driver routines for RALEIGH solver. Provide user friendly SciPy-style
   interfaces to RALEIGH abstract solver.

partial_hevp:
    Computes several eigenpairs of sparse real symmetric/Hermitian eigenvalue
    problems using either shift-invert or preconditioning technique.
    Requires MKL 10.3 or later (needs mkl_rt.dll on Windows, libmkl_rt.so on
    Linux).

partial_svd:
    Computes several largest singular values of a dense matrix and corresponding
    singular vectors. On CPU, performs better if MKL 10.3 or later is available.
    Can also run on GPU.
'''
