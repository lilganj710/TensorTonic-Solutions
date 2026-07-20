import numpy as np
from numpy.typing import ArrayLike


def pca_projection(X: ArrayLike, k: int) -> list[list[float]]:
    """
    Project data onto the top-k principal components.
        Instead of trying to implement "power iteration with deflation"
        manually, I'll use np.linalg
    """
    X = np.array(X)
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = (1 / (X.shape[0] - 1)) * X_centered.T @ X_centered
    eigvals, eigenvectors = np.linalg.eig(cov_matrix)
    eigval_ordering = np.flip(np.argsort(np.abs(eigvals)))
    eigenvectors = eigenvectors[:, eigval_ordering]
    return X_centered @ eigenvectors[:, :k]