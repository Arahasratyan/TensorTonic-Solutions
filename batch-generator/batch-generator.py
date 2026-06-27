import numpy as np


def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    N = len(y)
    X, y = np.array(X), np.array(y)
    ind = np.arange(0, N, dtype=np.int16)
    if rng:
        rng.shuffle(ind)
    else:
        np.random.shuffle(ind)
        
    i = -1
    for i in range(N // batch_size):
        yield X[ind[i * batch_size:(i + 1) * batch_size]], y[ind[i * batch_size:(i + 1) * batch_size]]


    if not drop_last and N % batch_size > 0:
        yield X[ind[(i + 1) * batch_size:N]], y[ind[(i + 1) * batch_size:N]]