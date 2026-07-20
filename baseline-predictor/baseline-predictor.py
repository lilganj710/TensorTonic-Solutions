import numpy as np
from numpy.typing import ArrayLike


def baseline_predict(ratings_matrix: ArrayLike, target_pairs: list[tuple[int, int]]) -> list[list[float]]:
    """
    Compute baseline predictions for each target_pairs using global mean and user/item biases.
        ratings_matrix[i, j] = the rating user i gave to item j
    """
    global_mean = np.sum(ratings_matrix) / np.count_nonzero(ratings_matrix)
    user_mean_ratings = (
        np.sum(ratings_matrix, axis=1)
        / np.count_nonzero(ratings_matrix, axis=1)
    )
    item_mean_ratings = (
        np.sum(ratings_matrix, axis=0)
        / np.count_nonzero(ratings_matrix, axis=0)
    )
    user_biases = user_mean_ratings - global_mean
    item_biases = item_mean_ratings - global_mean
    baseline_prediction = global_mean + user_biases[:, None] + item_biases
    return [baseline_prediction[*pair] for pair in target_pairs]