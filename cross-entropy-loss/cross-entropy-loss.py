import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    prob = np.array([x[y] for x, y in zip(y_pred, y_true)], dtype=np.float32)
    x = -np.mean(np.log(prob))

    return float(x)