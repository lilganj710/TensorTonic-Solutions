import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as ss


def get_priors(y_train: np.ndarray) -> np.ndarray:
    '''Return the prior probabilities for each class'''
    class_counts = np.bincount(y_train)
    return class_counts / len(y_train)

def get_gaussian_params(X_train: np.ndarray, y_train: np.ndarray,
                        var_eps: float=1e-9) -> tuple[np.ndarray, np.ndarray]:
    '''First group features (X_train) by class. Then, assume that within each class,
    each component of the features follows a Gaussian. Return means and vars of these Gaussians
    means[class, feature_idx] = the sample mean of the jth feature component in class
    vars[class, feature_idx] = the sample var of the jth feature component in class (no Bessel correction)
        For numerical stability, add var_eps to avoid 0 variance'''
    largest_class = max(y_train)
    means_by_class = np.zeros((largest_class+1, X_train.shape[1]), dtype=np.float64)
    vars_by_class = np.zeros((largest_class+1, X_train.shape[1]), dtype=np.float64)
    for class_num in range(largest_class+1):
        class_idxs = np.nonzero(y_train == class_num)
        cur_features = X_train[class_idxs]
        means_by_class[class_num] = np.mean(cur_features, axis=0)
        vars_by_class[class_num] = np.var(cur_features, axis=0)
    return means_by_class, vars_by_class+var_eps

def get_posteriors(
        X_test: np.ndarray, priors: np.ndarray,
        gaussian_params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    '''Use Bayes rule to get the (log) posteriors for each class, given each X_test sapmle.
    By assuming conditional independence among features (given class), we can use the chain rule,
    which allows us to get the posteriors via summing the log probs
    return.shape = (num classes, num test samples)'''
    gaussian_args = X_test.T[None, ...]
    means, vars = gaussian_params
    feature_log_probs = ss.norm.logpdf(gaussian_args, means[..., None], vars[..., None])
    log_probs_by_class = np.sum(feature_log_probs, axis=1)
    log_posteriors = np.log(priors[:, None]) + log_probs_by_class
    return log_posteriors

def get_predicted_classes(posteriors: np.ndarray, priors: np.ndarray) -> list[int]:
    '''Due to ambiguity in the problem statement, this isn't as simple as
    np.argmax(posteriors, axis=0). There's an unintuitive way for breaking
    ties, implemented here'''
    predicted_classes = [0 for _ in range(posteriors.shape[1])]
    for test_idx in range(posteriors.shape[1]):
        class_posteriors = posteriors[:, test_idx]
        sorted_classes = sorted(
            range(len(priors)),
            key=lambda class_idx: (class_posteriors[class_idx], priors[class_idx], -class_idx),
            reverse=True
        )
        predicted_classes[test_idx] = sorted_classes[0]
    return predicted_classes

def gaussian_naive_bayes(X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike) -> list[int]:
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
        Assumes class labels take values in {0, ..., n-1}
    Edge case: two posteriors have the same value. A test case suggests I should
        choose the larger of the two classes, not the smaller
        But then another test case says I should choose the smaller of the two?
        And the example graphic also says to pick the smaller?
    Tentatively, I have 2 tiebreakers
        First, tiebreak by prior probability. If still tied, then choose
        the smaller class
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    priors = get_priors(y_train)
    gaussian_params = get_gaussian_params(X_train, y_train)
    posteriors = get_posteriors(X_test, priors, gaussian_params)
    return get_predicted_classes(posteriors, priors)