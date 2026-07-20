import numpy as np
from numpy.typing import ArrayLike


def sigmoid(x: ArrayLike) -> np.ndarray:
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    return 1 / (1 + np.exp(-x))