import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    ans = []
    for pos in range(seq_length):
        i = np.array(range(d_model))
        vec = np.zeros(d_model)
        even_i = i[0::2]
        odd_i = i[0::2]
        vec[0::2] = PE_even = np.sin(pos/10000**(even_i/d_model))
        vec[1::2] = np.cos(pos/10000**(odd_i/d_model))

        ans.append(vec)

        
    return np.array(ans)
        