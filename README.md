## RALEIGH: RAL EIGensolver for real symmetric and Hermitian problems

RALEIGH is a Python implementation of the block Jacobi-conjugated gradients algorithm for computing several eigenpairs (eigenvalues and corresponding eigenvectors) of large scale real symmetric and Hermitian problems. 

### Key features

* Can be applied to both standard eigenvalue problem for a real symmetric or Hermitian matrix A and generalized eigenvalue problems for matrix pencils A - &lambda; B and A B - &lambda; I with positive definite real symmetric or Hermitian B.
* Can employ both known convergence improvement techniques for large sparse problems: shift-invert and preconditioning.
* Can also compute singular values and vectors, and is actually an especially efficient tool for Principal Component Analysis of dense data, owing to the high efficiency of matrix multiplications on modern multicore and GPU architectures.
* The user can specify wanted eigenvalues
	- on either margin of the spectrum (e.g. 5 on the left, 10 on the right)
	- of largest magnitude
	- on either side of a given real value
	- nearest to a given real value
	- If the number of eigenvalues needed is not known in advance (as is normally the case with PCA), the computation will continue until user-specified stopping criteria are satisfied (e.g. PCA approximation to the data is satisfactory).
* The core solver is written in terms of abstract vectors, owing to which it will work on any architecture provided that basic linear algebra operations on vectors are implemented (MKL and CUDA implementations are provided, in the absence of these Numpy algebra will be used).

### Dependencies

For best performance, install MKL 10.3 or later (or, on Windows, numpy+mkl). On Linux, the folder containing libmkl_rt.so must be in LD_LIBRARY_PATH. On Windows, the one containing mkl_rt.dll must be in PATH). Large sparse problems can only be solved if MKL is available, PCA and other dense problems can be solved without it.

To use GPU (which must be CUDA-enabled), you need NVIDIA GPU Computing Toolkit installed. On Linux, the folder containing libcudart.so must be in LD_LIBRARY_PATH.

### Basic usage

Subfolder _drivers_ contains user-friendly SciPy-like interfaces to core solver working in terms of NumPy and SciPy data objects. Subfolder _examples_ contains scripts illustrating their use, as well as a script illustrating basic capabilities of the core solver.

### Documentation

Basic usage of the package is briefly described in docstrings of scripts in folder _drivers_ (the best starting point) and example scripts. Advanced users will find the description of basic principles of RALEIGH's design in _core/solver.py_.

The mathematical and numerical aspects of the algorithm implemented by RALEIGH are described in the papers by E. E. Ovtchinnikov in J. Comput. Phys. 227:9477-9497 and SIAM Numer. Anal. 46:2567-2619.

### License

RALEIGH is released under 3-clause BSD licence.
