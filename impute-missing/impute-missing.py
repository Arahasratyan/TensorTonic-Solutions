import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X)

    nan_X = np.isnan(X)
    valid_X = np.logical_not(nan_X)

    if len(X.shape) == 1:
        if any(valid_X):
            if strategy == "mean":
                replace = np.mean(X[valid_X])
            else:
                replace = np.median(X[valid_X])
        else:
            replace = np.zeros(X.shape[0])

        X[nan_X] = replace
        return X
        
        


    if strategy=="mean":
        replace = np.array([np.mean(X[:, i][valid_X[:, i]]) if any(X[:, i][valid_X[:, i]]) else 0 for i in range(X.shape[1])])
    else:
        replace = np.array([np.median(X[:, i][valid_X[:, i]]) if any(X[:, i][valid_X[:, i]]) else 0 for i in range(X.shape[1])])
        
    for i in range(X.shape[1]):
        X[:, i][nan_X[:, i]] = replace[i]

    return X