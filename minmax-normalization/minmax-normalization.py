import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.array(X)
    if axis == 0:
        rsp = (1, -1)
    else:
        rsp = (-1, 1)
        
    min_X = np.min(X, axis=axis).reshape(rsp)
    max_X = np.max(X, axis=axis).reshape(rsp)

    denominator = max_X - min_X

    X = np.where(denominator != 0, (X - min_X) / denominator, 0.0)
    
    return X