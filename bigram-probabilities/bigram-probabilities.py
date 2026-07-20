import itertools as it
from collections import Counter


def bigram_probabilities(
        tokens: list[str]
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], float]]:
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
      Recall that "add-1 smoothing" comes from the rule of succession
    """
    vocab = set(tokens)
    bigram_counts = Counter([tuple(tokens[i:i+2]) for i in range(len(tokens)-1)])
    probs = {
        (w1, w2): (
            (bigram_counts[(w1, w2)] + 1)
            / (sum(bigram_counts[(w1, u)] for u in vocab) + len(vocab))
        )
        for w1, w2 in it.product(vocab, repeat=2)
    }
    return dict(bigram_counts), probs