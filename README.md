## RALEIGH: RAL EIGensolver for real symmetric and Hermitian problems

RALEIGH is a Python implementation of the block Jacobi-conjugated gradients algorithm for computing several eigenpairs (eigenvalues and corresponding eigenvectors) of large scale real symmetric and Hermitian problems. 

### Key features

* Can be applied to both standard eigenvalue problem for a real symmetric or Hermitian matrix A and generalized eigenvalue problems for matrix pencils A - &lambda; B or A B - &lambda; I with positive definite real symmetric or Hermitian B.
* Can employ either of the two known convergence improvement techniques for large sparse problems: shift-and-invert and preconditioning.
* Can also compute singular values and vectors, and is actually an efficient tool for Principal Component Analysis (PCA) of dense data of large size, owing to the high efficiency of matrix multiplications on modern multicore and GPU architectures.
* PCA capabilities include quick update of principal components after arrival of new data and incremental computation of principal components, dealing with one chunk of data at a time.
* For sparse matrices of large size (~10<sup>5</sup> or larger), RALEIGH's `partial_hevp` eigensolver is much faster than `eigsh` from SciPy. The table below shows the computation times in seconds for computing the smallest eigenvalue of 3 matrices from DNVS group of Suitesparse Matrix Collection and the smallest buckling load factor of 4 buckling problems on Intel(R) Xeon(R) CPU E3-1220 v3 @ 3.10GHz (the links to matrices' repositories can be found in `sparse_evp.py` and `buckling_evp.py` in subfolder `raleigh/examples`).

  | matrix | size | eigsh | partial_hevp |
  | - | - | - | - |
  | shipsec1 | 140874 | 240 | 6.9 |
  | shipsec5 | 179860 | 318 | 5.3 |
  | x104 | 108384 | 225 | 5.2 |
  | panel_buckle_d | 74383 | 26 | 1.4 |
  | panel_buckle_e | 144823 | 85 | 2.5 |
  | panel_buckle_f | 224522 | 135 | 3.8 |
  | panel_buckle_g | 394962 | 321 | 7.2 |

* Similarly, for large data (~10<sup>4</sup> samples with ~10<sup>4</sup> features or larger) that has large amount of redundancy, RALEIGH's `pca` function is considerably faster than `fit_ransform` method of scikit-learn and uses less memory. The computation times for PCA of 13233 images from Labeled Faces in the Wild (the link to LFW website can be found in `raleigh/examples/eigenimages/convert_lfw.py`) on the same CPU are:

  | components | scikit-learn pca | raleigh pca |
  | - | - | - |
  | 1000 | 128 | 53 |
  | 2000 | 180 | 101 |
  | 3000 | 288 | 165 |

* The core solver allows user to specify the number of wanted eigenvalues
	- on either margin of the spectrum (e.g. 5 on the left, 10 on the right)
	- of largest magnitude
	- on either side of a given real value
	- nearest to a given real value
* If the number of eigenvalues needed is not known in advance (as is normally the case with PCA), the computation will continue until user-specified stopping criteria are satisfied (e.g. PCA approximation to the data is satisfactory).
* The core solver is written in terms of abstract vectors, owing to which it will work on any architecture verbatim, as long as basic linear algebra operations on vectors are implemented. Currently, MKL and CUBLAS implementations are provided with the package, in the absence of these libraries NumPy algebra being used.

### Dependencies

For best performance, install MKL 10.3 or later. On Linux, the latest MKL can be installed by `pip install --user mkl`. On Windows, one can alternatively install numpy+mkl. If MKL is installed in any other way, make sure that, on Linux, the folder containing `libmkl_rt.so` is listed in `LD_LIBRARY_PATH`, and, on Windows, the one containing `mkl_rt.dll` is listed in `PATH`. If you do not know how to do it, then put `from raleigh.algebra import env` in your script and set `env.mkl_path` to that folder. Large sparse problems can only be solved if MKL is available, PCA and other dense problems can be tackled without it.

To use GPU (which must be CUDA-enabled), NVIDIA GPU Computing Toolkit needs to be installed. On Linux, the folder containing `libcudart.so` must be listed in `LD_LIBRARY_PATH`. At present, GPU can only be used for dense (SVD-related) problems.

### Package structure

#### Basic use subpackages

Subpackage `interfaces` contains user-friendly SciPy-like interfaces to core solver working in terms of NumPy and SciPy data objects. Subpackage `examples` contains scripts illustrating their use, as well as a script illustrating basic capabilities of the core solver.

#### Advanced use subpackages

Subpackage `algebra` contains NumPy, MKL and CUBLAS implementations of abstract vectors algebra. These can be used as templates for user's own implementations. Subpackage `core` contains the core solver implementation and related data objects definitions.

### Basic usage

To compute 10 eigenvalues closest to 0.25 of a sparse real symmetric or Hermitian matrix `A` in SciPy format:
```
from raleigh.interfaces.partial_hevp import partial_hevp
lmd, x, status = partial_hevp(A, which=10, sigma=0.25)
# lmd : eigenvalues
# x : eigenvectors
# status : execution status
```
To compute 10 smallest eigenvalues of a sparse positive definite real symmetric or Hermitian matrix `A` using its incomplete LU-factorization as the preconditioner:
```
from raleigh.interfaces.partial_hevp import partial_hevp
from raleigh.algebra.sparse_mkl import IncompleteLU as ILU
T = ILU(A)
T.factorize()
lmd, x, status = partial_hevp(A, which=10, T=T)
```
To compute 10 lowest buckling load factors &alpha; of the buckling problem (K + &alpha; Ks)v = 0 with stiffness matrix K and stress stiffness matrix Ks using load factor shift 1.0:
```
from raleigh.interfaces.partial_hevp import partial_hevp
alpha, v, status = partial_hevp(K, Ks, buckling=True, sigma=-1.0, which=10)
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

Documenting RALEIGH is still _work in progress_ at the moment due to the large size of the package and other commitments of the author. Basic usage of the package is briefly described in the docstrings of modules in `interfaces` and `examples`. Advanced users will find the description of basic principles of RALEIGH's design in `core` module `solver`.

The mathematical and numerical aspects of the algorithm implemented by RALEIGH are described in the papers by E. E. Ovtchinnikov in **J. Comput. Phys. 227:9477-9497** and **SIAM Numer. Anal. 46:2567-2619**. A Fortran90 implementation of this algorithm was used in a paper on Topology Optimization by P.D. Dunning, E. Ovtchinnikov, J. Scott and H.A. Kim in **International Journal for Numerical Methods in Engineering 107 (12), 1029-1053** (the four buckling problems mentioned above were used for the performance testing and comparisons with ARPACK). A pre-release version of RALEIGH was used in a paper by A. Liptak, G. Burca, J. Kelleher, E. Ovtchinnikov, J. Maresca and A. Horner in **Journal of Physics Communications 3 (11), 113002**.

### Feedback

Please use [GitHub issue tracker](https://github.com/evgueni-ovtchinnikov/raleigh/issues) or send an e-mail to Evgueni to report bugs and request features.

### License

RALEIGH is released under 3-clause BSD licence.
