import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p, y = np.asarray(p).flatten(), np.asarray(y).flatten()
    dice = (2*sum(p*y) + eps)/(sum(p)+sum(y) + eps)
    return 1 - dice