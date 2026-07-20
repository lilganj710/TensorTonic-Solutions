import numpy as np
from numpy.typing import ArrayLike


def confusion_matrix_norm(
        y_true: ArrayLike, y_pred: ArrayLike,
        num_classes: int | None=None, normalize: str='none') -> np.ndarray:
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1
    confusion_matrix_idxs = (num_classes * y_true + y_pred).astype(np.int64)
    confusion_matrix_counts = np.bincount(confusion_matrix_idxs, minlength=num_classes**2)
    confusion_matrix = np.reshape(
        confusion_matrix_counts, (num_classes, num_classes)
    ).astype(np.float64)
    if normalize == 'none':
        pass
    elif normalize == 'true':
        confusion_matrix /= np.sum(confusion_matrix, axis=1)[:, None]
    elif normalize == 'pred':
        confusion_matrix /= np.sum(confusion_matrix, axis=0)
    elif normalize == 'all':
        confusion_matrix /= np.sum(confusion_matrix)
    else:
        raise ValueError(f'invalid {normalize=}')
    return confusion_matrix