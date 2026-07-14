import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """

    thresholds = np.array([np.inf] + sorted(list(set(y_score)), reverse=True))
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    tpr, fpr = [], []
    
    for threshold in thresholds:
        y_score_pr = y_score >= threshold
        TP = np.sum((y_score_pr==1) & (y_true==1))
        FP = np.sum((y_score_pr==1) & (y_true==0))
        TN = np.sum((y_score_pr==0) & (y_true==0))
        FN = np.sum((y_score_pr==0) & (y_true==1))
        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN))
        
    
    return np.array(fpr), np.array(tpr), thresholds
    