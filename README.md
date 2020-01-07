## RALEIGH: RAL EIGensolver for real symmetric and Hermitian problems

RALEIGH is a Python implementation of the block Jacobi-conjugated gradients algorithm for computing several eigenpairs (eigenvalues and corresponding eigenvectors) of large scale real symmetric and Hermitian problems. 

### Key features

* Can be applied to both standard eigenvalue problem for a real symmetric or Hermitian matrix A and generalized eigenvalue problems for matrix pencils A - &lambda; B or A B - &lambda; I with positive definite real symmetric or Hermitian B.
* Can employ either of the two known convergence improvement techniques for large sparse problems: shift-and-invert and preconditioning.
* Can also compute singular values and vectors, and is actually an especially efficient tool for Principal Component Analysis (PCA) of dense data of large size (more than 10000 samples with more than 10000 features each), owing to the high efficiency of matrix multiplications on modern multicore and GPU architectures.
* The user can specify the number of wanted eigenvalues
	- on either margin of the spectrum (e.g. 5 on the left, 10 on the right)
	- of largest magnitude
	- on either side of a given real value
	- nearest to a given real value
* If the number of eigenvalues needed is not known in advance (as is normally the case with PCA), the computation will continue until user-specified stopping criteria are satisfied (e.g. PCA approximation to the data is satisfactory).
* PCA capabilities include quick update of principal components after arrival of new data and incremental computation of principal components, dealing with one chunk of data at a time.
* The core solver is written in terms of abstract vectors, owing to which it will work on any architecture verbatim, provided that basic linear algebra operations on vectors are implemented. Currently, MKL and CUBLAS implementations are provided with the package, in the absence of these libraries NumPy algebra being used.

### Dependencies

For best performance, install MKL 10.3 or later (or, on Windows, numpy+mkl). On Linux, the folder containing libmkl\_rt.so must be listed in LD\_LIBRARY\_PATH. On Windows, the one containing mkl\_rt.dll must be listed in PATH. Large sparse problems can only be solved if MKL is available, PCA and other dense problems can be dealt with without it.

To use GPU (which must be CUDA-enabled), NVIDIA GPU Computing Toolkit needs to be installed. On Linux, the folder containing libcudart.so must be listed in LD\_LIBRARY\_PATH.

### Package structure

#### Basic use subpackages

Subpackage _interfaces_ contains user-friendly SciPy-like interfaces to core solver working in terms of NumPy and SciPy data objects. Subpackage _examples_ contains scripts illustrating their use, as well as a script illustrating basic capabilities of the core solver.

#### Advanced use subpackages

Subpackage _algebra_ contains NumPy, MKL and CUBLAS implementations of abstract vectors algebra. These can be used as templates for user's own implementations. Subpackage _core_ contains the core solver implementation and related data objects definitions.

### Basic usage

To compute 10 eigenvalues closest to 0.25 of a sparse real symmetric or Hermitian matrix `A` in SciPy format:
```
from raleigh.interfaces.partial_hevp import partial_hevp
lmd, x, status = partial_hevp(A, which=10, sigma=0.25)
# lmd : eigenvalues
# x : eigenvectors
# status : execution status
```
To compute 10 smallest eigenvalues of a sparse positive definite real symmetric or Hermitian matrix `M` using its incomplete LU-factorization as the preconditioner:
```
from raleigh.algebra.sparse_mkl import IncompleteLU as ILU
T = ILU(A)
T.factorize()
lmd, x, status = partial_hevp(A, which=10, T=T)
```
To compute 100 principal components for the dataset represented by the 2D matrix `A` with data samples as rows:
```
from raleigh.interfaces.pca import pca
mean, trans, comps = pca(A, npc=100)
# mean : the average of data samples
# trans : transformed (reduced features) data set
# comps : the matrix with principal components as rows
```
To compute a number of principal components sufficient to approximate `A` with 5% tolerance to the relative PCA error (the ratio of the Frobenius norm of `trans*comps - A_s` to that of `A_s`, where the rows of `A_s` are the original data samples shifted by `mean`):
```
mean, trans, comps = pca(A, tol=0.05)
```
To quickly update `mean`, `trans` and `comps` taking into account new data `A_new`:
```
mean, trans, comps = pca(A_new, have=(mean, trans, comps))
```
To compute 5% accuracy PCA approximation incrementally by processing 1000 data samples at a time:
```
mean, trans, comps = pca(A, tol=0.05, batch_size=1000)
```

### Documentation

Basic usage of the package is briefly described in the docstrings of modules in _interfaces_ and _examples_. Advanced users will find the description of basic principles of RALEIGH's design in _core_ module _solver_.

The mathematical and numerical aspects of the algorithm implemented by RALEIGH are described in the papers by E. E. Ovtchinnikov in J. Comput. Phys. 227:9477-9497 and SIAM Numer. Anal. 46:2567-2619.

### Issues

Please use [GitHub issue tracker](https://github.com/evgueni-ovtchinnikov/raleigh/issues) or send an e-mail to Evgueni to report bugs and request features.

### License

RALEIGH is released under 3-clause BSD licence.
