import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _loss_gradient(X, y_pr, y, N):
    """Binary Cross-Entropy loss gradients for w and b."""
    grad_w = np.dot(X.T, (y_pr - y))/N
    grad_b = np.mean(y_pr-y)
    return grad_w, grad_b
    
    

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    w = np.zeros(X.shape[1])
    b = 0.0

    X, y = np.array(X), np.array(y)
    N = len(y)
    
    for i in range(steps): 
        y_pr = _sigmoid(np.dot(X, w)+b)
        grad_w, grad_b = _loss_gradient(X, y_pr, y, N)
        w -= lr*grad_w
        b -= lr*grad_b
    return (w, b)
        
        
    
    