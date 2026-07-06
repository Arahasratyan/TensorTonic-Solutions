import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    if rng is None:
        rng = np.random.default_rng()

    dropout_pattern = rng.random(x.shape) >= p
    output = np.where(dropout_pattern, x * 1 / (1 - p), 0)
    
    return output, dropout_pattern * 1 / (1 - p)