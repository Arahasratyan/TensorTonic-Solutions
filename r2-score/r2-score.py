import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if all(y_true==y_pred):
        return 1
    unexplained_variance = np.sum((y_pred-y_true)**2)
    total_variance = np.sum((y_true-np.mean(y_true))**2)
    if total_variance == 0:
        return 0

    return 1 - unexplained_variance/total_variance
    