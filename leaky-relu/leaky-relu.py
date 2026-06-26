import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x, np.float32)
    x[x<0] = alpha * x[x<0]
    return x