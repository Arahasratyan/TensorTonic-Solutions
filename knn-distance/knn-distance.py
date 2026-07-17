import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    if len(X_train.shape) != 2:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        
    pairwise_distances = ((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])**2).sum(axis=2)

    ans = pairwise_distances.argsort()[:, 0:k]
    actual_neighbors_found = ans.shape[1]
    if actual_neighbors_found < k:
        padding_size = k - actual_neighbors_found
        ans = np.pad(ans, ((0, 0), (0, padding_size)), mode='constant', constant_values=-1)
        
    return ans
        
    return ans