import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    x, W, b = np.asarray(x), np.asarray(W), np.asarray(b)

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    H_out = H-KH + 1
    W_out = W_in - KW + 1

    y = np.zeros((N, C_out, H_out, W_out))

    for u in range(KH):
        for v in range(KW):
            x_slice = x[:, :, u:u+H_out, v:v+W_out]
            
            y += np.sum(x_slice[:, None, :, :, :] * W[None, :, :, u, v, None, None], axis=2)
            
            # x_sliced:    (N,    1,    C_in,  H_out,  W_out)
            # W_expanded:  (1,  C_out,  C_in,    1,      1)
            #               |     |       |      |       |
            #               v     v       v      v       v
            # Broadcasts   (N,  C_out,  C_in,  H_out,  W_out)

    y += b[:, None, None]
    # y:           (N,  C_out,  H_out,  W_out)
    # b_expanded:       (C_out,    1,      1)  

    return y
            
                    
    