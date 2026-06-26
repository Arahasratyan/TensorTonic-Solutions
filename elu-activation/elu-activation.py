import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.array(x, dtype=np.float32)
    x[x<0] = alpha * (np.exp(x[x<0])-1)
    return list(x)