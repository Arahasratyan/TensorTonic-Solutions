import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x)
    s = np.std(x, ddof=1)

    t = (np.mean(x) - mu0)/(s/np.sqrt(len(x)))
    return t