import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.array(x)
    x[x<=0] = 0
    return x