import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    if len(tpr) == len(fpr) and len(tpr) > 1:
        return np.trapezoid(tpr, x=fpr)
        
        