import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x, gamma, beta = np.asarray(x), np.asarray(gamma), np.asarray(beta)
    axs = 0 if len(x.shape) == 2 else (0, 2, 3)

    mu = np.mean(x, axis=axs)
    variance = np.var(x, axis=axs)
    if len(x.shape) != 2:
        mu = mu[np.newaxis, :, np.newaxis, np.newaxis]
        variance = variance[np.newaxis, :, np.newaxis, np.newaxis]
        gamma = gamma[np.newaxis, :, np.newaxis, np.newaxis]
        beta = beta[np.newaxis, :, np.newaxis, np.newaxis]
    x = (x - mu)/np.sqrt(variance + eps)
    y = gamma*x + beta
    return y

        
    
    