import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x, p = np.array(x, np.float32), np.array(p, np.float32)
    if np.abs(np.sum(p) - 1) > 0.01:
        raise ValueError("ValueError")
    return np.sum(x*p)
