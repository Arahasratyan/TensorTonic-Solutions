import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x)
    pmf = p**x * (1-p)**(1-x)
    
    mean = p
    var = p*(1-p)


    return pmf, mean, var
    