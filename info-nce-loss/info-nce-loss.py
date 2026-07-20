import numpy as np
from scipy.special import softmax

def info_nce_loss(Z1: ArrayLike, Z2: ArrayLike, temperature: float=0.1) -> float:
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    similarity_matrix = (Z1 @ Z2.T) / temperature
    softmaxes = softmax(similarity_matrix, axis=1)
    return -np.mean(np.log(np.diag(softmaxes)))