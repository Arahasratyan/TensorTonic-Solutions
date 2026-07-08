import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    
    if not matrix:
        return None
    N = len(matrix)
    for i in matrix:
        if not isinstance(i, list) or len(i) != N:
            return None
    matrix = np.asarray(matrix)
    if len(matrix.shape)>2:
        return None

        
    e = np.linalg.eigvals(matrix) 
    real, imag = e.real, e.imag 
    ind = np.lexsort((real, imag))
    e = np.array([real[i] + imag[i]*1j for i in ind])
    return e