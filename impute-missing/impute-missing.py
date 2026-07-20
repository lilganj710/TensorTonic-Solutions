import numpy as np
from numpy.typing import ArrayLike


def impute_missing(X: ArrayLike, strategy='mean') -> np.ndarray:
    """
    Fill NaN values in each feature column using column mean or median.
        Edge case: col is all NaN...fill with 0
    Return the filled array
    """
    X = np.array(X)
    if strategy not in ['mean', 'median']:
        raise ValueError(f'invalid {strategy=}')
    imputation_func = np.nanmean if strategy == 'mean' else np.nanmedian
    X = np.where(np.all(np.isnan(X), axis=0), 0, X)
    vals_to_impute = imputation_func(X, axis=0)
    imputed_arr = np.where(np.isnan(X), vals_to_impute, X)
    return imputed_arr