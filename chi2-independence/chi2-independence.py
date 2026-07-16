import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    C = np.asarray(C)
    r = np.sum(C, axis=0)
    c = np.sum(C, axis=1)
    
    E = np.outer(c, r)/np.sum(C)
    return np.sum((C - E)**2 / E), E