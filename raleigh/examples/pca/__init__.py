'''Example and utility scripts.

pca_simple.py:
    Computes a number of principal components of a dataset.
    If sklearn is installed, compares with sklearn.decomposition.PCA.

pca_smart.py:
    Computes PCA approximation of specified accuracy.
    If sklearn is installed, compares with restarted sklearn.decomposition.PCA.

pca_update.py:
    Performs PCA on a chunk of data, then addds more data and updates principal 
    components.

incremental_pca.py:
    Incremental Principal Component Analysis demo.
    Principal components are computed incrementally, processing a number 
    (batch size) of data samples at a time.
    For comparison, the same amount of principal components is computed also by 
    scikit-learn IncrementalPCA.

interactive_pca.py:
    Computes principal components of a dataset until stopped by the user.

generate_matrix.py:
   Generates a data matrix from randomly generated singular vectors and 
   singular values of user-controlled behaviour.
   If run as a script, the generated matrix is saved to data.npy, and singular
   values and vectors to svd.npz.
   Alternatively, from generate_matrix import generate, after which the call of
   generate(m, n, rank, dtype, scale, alpha, pca) will return the matrix,
   singular values and left and right singular vectors.

'''
