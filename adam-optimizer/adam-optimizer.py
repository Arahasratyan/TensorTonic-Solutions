import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    grad, m, v, param = np.asarray(grad), np.asarray(m), np.asarray(v), np.asarray(param)
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad**2
    m_ = m/(1-beta1**t)
    v_ = v/(1-beta2**t)
    param = param - lr * m_/(np.sqrt(v_) + eps)
    return param, m, v