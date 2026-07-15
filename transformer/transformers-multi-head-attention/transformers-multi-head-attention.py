import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    d_k = d_model // num_heads  

    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    Q_proj = np.dot(Q, W_q).reshape(batch, seq_len_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_proj = np.dot(K, W_k).reshape(batch, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_proj = np.dot(V, W_v).reshape(batch, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)

    # Shape: (batch, num_heads, seq_len_q, seq_len_k)
    att_scores = np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attention_weights = softmax(att_scores, axis=-1)
    
    # Shape: (batch, num_heads, seq_len_q, d_k)
    context = np.matmul(attention_weights, V_proj)
    
    context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len_q, d_model)
    
    output = np.dot(context, W_o)
    
    return output