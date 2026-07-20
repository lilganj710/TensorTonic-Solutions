import numpy as np
from numpy.typing import ArrayLike


def get_class_split(
        X: np.ndarray, y: np.ndarray, test_proportion: float,
        rng: np.random.Generator, class_num: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''For a particular class label, do the train/test split, respecting
    the test_size proportion as closely as possible
    Return (X_train, X_test, y_train, y_test) for this class
    
    It seems like "shuffle within each class using the rng" isn't actually what the test cases do?
        Hints say to use rng.shuffle() instead of rng.permutation()...yet the output remains the same
        Now, I'm left to guess what random shuffling they actually used
        Maybe rng.shuffle() directly on the cur_X? Apparently not
    
    After some experimentation, it seems like after shuffling, the first part is used
        for test and the second is used for train (rather than using first for train and second for test)
        Moreover, they seem to be sorted by original ordering'''
    cur_class_indices = rng.permutation(np.nonzero(y == class_num)[0])
    test_size = int(round(test_proportion * len(cur_class_indices)))
    train_size = len(cur_class_indices) - test_size
    test_indices = np.sort(cur_class_indices[:test_size])
    train_indices = np.sort(cur_class_indices[test_size:])
    return (
        X[train_indices], X[test_indices],
        np.full(train_size, class_num), np.full(test_size, class_num)
    )


def stratified_split(
        X: ArrayLike, y: ArrayLike, test_size: float=0.2,
        rng: np.random.Generator | None=None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split features X and labels y into train/test while preserving class proportions.
        According to the prompt, I should use the rng to "shuffle within each class"
        And according to the example, rounding should be used to extract the test samples from each class
    Return (X_train, X_test, y_train, y_test)
    """
    if rng is None:
        print('rng not provided')
        rng = np.random.default_rng()
    X = np.array(X)
    y = np.array(y)
    unique_classes = np.unique(y)
    class_splits = [
        get_class_split(X, y, test_size, rng, class_num)
        for class_num in unique_classes
    ]
    return [
        np.concatenate(data_by_class, axis=0)
        for data_by_class in list(zip(*class_splits))
    ]