import numpy as np
from numpy.typing import ArrayLike


def get_priors(y_train: np.ndarray) -> np.ndarray:
    '''For each class in y_train, get the prior probabilities
    return.shape = (num classes,)'''
    return np.bincount(y_train) / len(y_train)


def get_conditional_feature_probs(
        X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    '''From the training set, extract the conditional probabilities of each
    feature being = 1, given each class
    As described, use Laplace-smoothed probabilities (rule of succession)
    return.shape = (num_features, num_classes)'''
    raw_probs = np.array([
        np.bincount(y_train, weights=feature_vals)
        for feature_vals in X_train.T
    ])
    class_counts = np.bincount(y_train)
    return (raw_probs + 1) / (class_counts + 2)


def naive_bayes_bernoulli(
        X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike) -> np.ndarray:
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
        X_train.shape = (num training samples, num_features)
        X_test.shape = (num testing samples, num_features)
    return.shape = (num testing samples, num classes)
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    prior_probs = get_priors(y_train)
    conditional_feature_probs = get_conditional_feature_probs(
        X_train, y_train)[None, ...]
    test_probs_each_feature = np.where(
        X_test[..., None] == 1,
        conditional_feature_probs, 1 - conditional_feature_probs
    )
    test_conditional_probs = np.prod(test_probs_each_feature, axis=1)
    return np.log(test_conditional_probs * prior_probs)