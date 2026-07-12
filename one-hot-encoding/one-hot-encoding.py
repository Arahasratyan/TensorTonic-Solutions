import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y)
    if not num_classes:
        num_classes = np.max(y) + 1
    one_hot_vector = np.eye(num_classes, dtype=np.float32)
    ans = np.array([one_hot_vector[i] for i in y])
    return ans