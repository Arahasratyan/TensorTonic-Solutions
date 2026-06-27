import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    
    if not max_len:
        max_len = max(map(len, seqs))
    L = len(seqs)
    ans = np.full((L, max_len), pad_value)
    for i in range(L):
        seq = seqs[i]
        l = len(seq)
        l = min(l, max_len)
        ans[i][:l] = np.array(seq[:l])
    return ans