import itertools as it
import numpy as np


def pool_adjacent_violators_algo(label_values: list[float]) -> list[float]:
    '''Start by assuming that y = c. Then, if there's a monotonicity violation
    (y_i > y_{i-1}), average them together
    I should be able to do this quickly by maintaining a monotone stack of (avg val, num elements)
    
    Proof of correctness comes from convex optimization machinery that I could probably
    work through given enough time. For now though, I'll accept that this algo is
    pretty intuitive'''
    regression_stack: list[tuple[float, int]] = [(label_values[0], 1)]
    for y_i in label_values[1:]:
        regression_stack.append((y_i, 1))
        while (
            len(regression_stack) > 1
            and regression_stack[-1][0] < regression_stack[-2][0]
        ):
            prev_val, prev_count = regression_stack.pop()
            penultimate_val, penultimate_count = regression_stack.pop()
            avg_val = (
                (prev_val * prev_count + penultimate_val * penultimate_count)
                / (prev_count + penultimate_count)
            )
            new_count = prev_count + penultimate_count
            regression_stack.append((avg_val, new_count))
    isotonic_components = [[val] * count for val, count in regression_stack]
    return list(it.chain.from_iterable(isotonic_components))


def calibrate_isotonic(
        cal_labels: list[int], cal_probs: list[float], new_probs: list[float]) -> list[float]:
    """
    Apply isotonic regression calibration.
        Given cal_labels, cal_probs, fit a monotone increasing curve using the
        given quadratic objective. This can be done with the Pool Adjacent Violators Algorithm
    Then, use the curve to transform new_probs
    """
    probs_argsort = sorted(range(len(cal_probs)), key=cal_probs.__getitem__)
    sorted_labels = [cal_labels[i] for i in probs_argsort]
    cal_probs = [cal_probs[i] for i in probs_argsort]
    isotonic_regression = pool_adjacent_violators_algo(sorted_labels)
    return np.interp(new_probs, cal_probs, isotonic_regression).tolist()