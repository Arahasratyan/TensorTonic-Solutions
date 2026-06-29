import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y, dtype=np.float32)
    N = len(y)
    values, counts = np.unique(y, return_counts=True)

    prob = np.array(counts)/N
    
    return -np.sum(prob * np.log2(prob))
    