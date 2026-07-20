import numpy as np
from collections import Counter
import math


def get_inverse_document_frequencies(
        query_tokens: list[str], doc_counts: list[dict[str, int]]) -> dict[str, float]:
    '''Get the IDF of each query token using the given formula
    Return a dict of (query token, IDF(token))'''
    document_frequencies = [
        sum(doc_count[q_token] > 0 for doc_count in doc_counts)
        for q_token in query_tokens
    ]
    inverse_doc_freqs = [
        np.log((len(doc_counts) - df_t + 0.5) / (df_t + 0.5) + 1)
        if df_t > 0 else 0
        for df_t in document_frequencies
    ]
    return {
        q_token: idf for q_token, idf in
        zip(query_tokens, inverse_doc_freqs, strict=True)
    }
    

def bm25_score(
        query_tokens: list[str], docs: list[list[str]],
        k1: float=1.2, b: float=0.75) -> np.ndarray:
    """
    Returns numpy array of BM25 scores for each document.
    """
    doc_counts = [Counter(doc) for doc in docs]
    avg_doc_len = np.mean([len(doc) for doc in docs])
    inverse_doc_freqs = get_inverse_document_frequencies(query_tokens, doc_counts)
    scores_each_doc: list[float] = [0 for _ in docs]
    for doc_idx in range(len(docs)):
        cur_doc_len = len(docs[doc_idx])
        scores_each_doc[doc_idx] = sum(
            inverse_doc_freqs[q_token]
            * (doc_counts[doc_idx][q_token] * (k1 + 1))
            / (doc_counts[doc_idx][q_token] + k1 * (1 - b + b*cur_doc_len/avg_doc_len))
            for q_token in query_tokens
        )
    return np.array(scores_each_doc)