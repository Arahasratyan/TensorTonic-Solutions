import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    
    x = np.array(x, np.float32)
    x = x - np.max(x)
    exp_arr = np.exp(x)
    if len(x.shape) == 2:
        exp_sum = np.sum(exp_arr, axis=1).reshape(exp_arr.shape[0], -1)
        return exp_arr / exp_sum
    exp_sum = np.sum(exp_arr)
    return exp_arr / exp_sum