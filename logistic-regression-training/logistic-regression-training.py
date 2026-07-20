import numpy as np
from typing import Callable, Any


def testing_decorator(func: Callable) -> Callable:
    '''View the test cases that are being passed into the given func'''
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        '''Prints the testcase, returns the original func return'''
        print(f'{args=}')
        res = func(*args, **kwargs)
        return res
    return wrapper


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(
        X: np.ndarray, y: np.ndarray,
        lr: float=0.1, steps: int=1000
        ) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent given the learning
    rate and the number of steps. Return (w, b).
    X.shape should be (N, d), where N is num observations, d is dimension
    """
    # Write code here
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    w = np.ones(X.shape[1])
    for _step_num in range(steps):
        Xw = X @ w
        sigmoids = _sigmoid(Xw)
        gradient_factors = -(
            (y/sigmoids - (1-y)/(1-sigmoids)) * sigmoids**2 * np.exp(-Xw)
        )
        gradients = gradient_factors[:, None] * X
        avg_gradient = np.mean(gradients, axis=0)
        w -= lr * avg_gradient
    return w[:-1], w[-1]