import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    Fully vectorized version.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    unique_scores = np.unique(y_score)[::-1]
    thresholds = np.r_[np.inf, unique_scores]
    
    y_score_pr = y_score[:, None] >= thresholds[None, :]
    
    TP = np.sum(y_score_pr & (y_true[:, None] == 1), axis=0)
    FP = np.sum(y_score_pr & (y_true[:, None] == 0), axis=0)
    
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    tpr = TP / P if P > 0 else np.zeros_like(TP, dtype=float)
    fpr = FP / N if N > 0 else np.zeros_like(FP, dtype=float)
    
    return fpr, tpr, thresholds
    