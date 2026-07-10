import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X)
    if len(X.shape) != 2 or X.shape[0] == 1:
        return None
        
    mu = np.mean(X, axis=0)
    X = X - mu
    cov = 1/(X.shape[0]-1) * X.T @ X
    return cov
    