import numpy as np
from numpy.typing import ArrayLike


def get_gini_impurity(labels: np.ndarray) -> float:
    '''Given integer labels, return the Gini impurity
    Wikipedia: Gini impurity = sum(p_i(1 - p_i), 1 <= i <= n) = the probability
    of choosing a random class and then mislabeling it
        Where both choice & labelling is done according to the distribution of classes'''
    empirical_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(empirical_probs**2)

def decision_tree_split(X: ArrayLike, y: ArrayLike) -> tuple[int, float]:
    """
    Find the best feature and threshold to split the data.
    """
    X = np.array(X)
    y = np.array(y)
    parent_gini = get_gini_impurity(y)
    max_gain = 0
    best_feature, best_threshold = -1, -1
    for feature_idx in range(X.shape[1]):
        cur_feature_vals = X[:, feature_idx]
        cur_sorted_unique = np.unique(cur_feature_vals)
        cur_thresholds = (cur_sorted_unique[1:] + cur_sorted_unique[:-1]) / 2
        for threshold in cur_thresholds:
            left_split = (cur_feature_vals <= threshold)
            left_split_size = np.sum(left_split)
            gini_left = get_gini_impurity(y[left_split])
            gini_right = get_gini_impurity(y[~left_split])
            gini_split = (
                left_split_size / X.shape[0] * gini_left
                + (X.shape[0] - left_split_size) / X.shape[0] * gini_right
            )
            gini_gain = parent_gini - gini_split
            if gini_gain > max_gain:
                max_gain = gini_gain
                best_feature, best_threshold = feature_idx, threshold
    return best_feature, best_threshold