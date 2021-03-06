# ChangeLog

## v1.3

- Added buckling mode to `partial_hevp` interface and buckling demo to examples.
- Added MKL installation instructions and automatic search for pip-installed MKL.

## v1.2

On the development side, this release mostly adds features that facilitate the user's development of interactive truncated SVD and PCA code. The use of these new features is demonstrated by the new icompute_eigenimages.py example script.

Presentation-wise, the users are likely to appreciate basic usage examples added to README.md, which are much more simple than the example scripts of the package.

## v1.1

This release mostly introduces substantial changes in computing Lower Rank Approximations (LRA) and Principal Component Analysis (PCA).
- Files/folders rearranged.
- PCA/LRA update implemented.
- Incremental PCA/LRA implemented.
- Comparisons with sklearn pca added to some examples.
- Simple usage examples via doctest added in pca() docstring.
- A simple alternative to docopt implemented in example scripts to avoid forcing the user to install it.

## v1.0

First release
