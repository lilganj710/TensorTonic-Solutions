import numpy as np
from numpy.typing import ArrayLike


def preprocess(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    '''Given two array-likes, preprocess them into np arrays
    sorted by score in reverse order'''
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    score_argsort = np.argsort(y_score)[::-1]
    return y_true[score_argsort], y_score[score_argsort]


def get_prefix_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    '''return[i] = an array of length 4 containing confusion matrix entries
    from y_true[:i] and y_pred[:i] (where indices are inclusive)
    In the order 00, 01, 10, 11 --> TN, FP, FN, TP'''
    entry_indicators = np.zeros((len(y_true), 4), dtype=np.int64)
    entry_indicators[np.arange(len(y_true)), 2 * y_true + y_pred] = 1
    return np.cumsum(entry_indicators, axis=0)


def roc_curve(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve from binary labels and scores.
        As described, treat each unique score as a threshold
            where x >= threshold --> classify true
        Then, given each threshold, get the TPR and FPR.
        This yields an (FPR, TPR) point on the Receiver Operating Characteristic curve
    Return the (FPRs, TPRs, thresholds) characterizing the ROC curve

    Note that the naive double-loop approach won't work. N <= 1e6 --> O(N^2) is too slow
        I guess I need prefix & suffix confusion matrices. At each threshold,
        corresponding matrices can be added together, then, the appropriate rates
        can be computed
    """
    y_true, y_score = preprocess(y_true, y_score)
    prefix_confusions = get_prefix_confusion_matrices(
        y_true, np.ones_like(y_score, dtype=np.int64)
    )
    prefix_confusions = np.vstack((np.zeros(4, dtype=np.int64), prefix_confusions))
    suffix_confusions = np.flip(
        get_prefix_confusion_matrices(
            np.flip(y_true), np.zeros_like(y_score, dtype=np.int64)),
        axis=0
    )
    suffix_confusions = np.vstack((suffix_confusions, np.zeros(4, dtype=np.int64)))
    joint_confusions = prefix_confusions + suffix_confusions
    true_neg, false_pos, false_neg, true_pos = joint_confusions.T
    false_positive_rates = false_pos / (false_pos + true_neg)
    true_positive_rates = true_pos / (true_pos + false_neg)
    thresholds = np.insert(y_score, 0, np.inf)
    _, unique_threshold_idxs = np.unique(thresholds[::-1], return_index=True)
    return (
        false_positive_rates[unique_threshold_idxs],
        true_positive_rates[unique_threshold_idxs],
        thresholds[unique_threshold_idxs]
    )