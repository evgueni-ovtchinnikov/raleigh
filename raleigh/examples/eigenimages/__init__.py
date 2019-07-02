'''An example of RALEIGH application to computing eigenimages (Principal
   Components of 2D images sets).

convert_lfw.py:
    Reads images from an lfw-style folder (see http://vis-www.cs.umass.edu/lfw),
    optionally processes them and saves in .npy file.
    === MUST BE RUN FIRST ===

compute_eigenimages.py:
    Computes Principal Components for a set of 2D images in small portions until
    stopped by user's entering 'n' in answer to 'more?' or the error of PCA
    approximation falls below the required tolerance.

show_errors.py:
    Compares images with their PCA approximations computed by the script
    compute_eigenimages.py.
'''
