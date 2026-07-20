import numpy as np
from scipy.stats import gmean


def bleu_score(candidate: list[str], reference: list[str], max_n: int) -> float:
    """
    Compute the BLEU score for a candidate translation.
        For n in [1, max_n], consider a sliding window of length n. Get the overlap
        between the candidate[window] and reference[window]. This gets the precision
        Then, as described, take the geometric mean of precisions, multiply by brevity penalty
    Edge cases:
        If any precision is 0, return 0
    """
    if len(candidate) == 0:
        return 0
    window_precisions: list[float] = []
    for window_len in range(1, max_n+1):
        window_matches = 0
        for window_start in range(len(candidate)-window_len+1):
            candidate_window = candidate[window_start:window_start+window_len]
            reference_window = reference[window_start:window_start+window_len]
            window_matches += int(candidate_window == reference_window)
        cur_precision = window_matches / (len(candidate)-window_len+1)
        if cur_precision == 0:
            return 0
        window_precisions.append(cur_precision)
    brevity_penalty = np.exp(min(1 - len(reference) / len(candidate), 0))
    return brevity_penalty * gmean(window_precisions)