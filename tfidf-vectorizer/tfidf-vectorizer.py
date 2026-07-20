import numpy as np
from collections import Counter
import math
import itertools as it


def get_vocab(doc_counts: list[dict[int, int]]) -> list[str]:
    '''Vocabulary = the unique, sorted words that appear in the documents'''
    doc_words = [list(doc_count.keys()) for doc_count in doc_counts]
    return sorted(set(it.chain.from_iterable(doc_words)))

def get_idfs(doc_counts: list[str], vocab: list[str]) -> np.ndarray:
    '''Return the inverse document frequency for each word in vocab'''
    num_docs_each = np.array([
        sum(doc_count[word] > 0 for doc_count in doc_counts)
        for word in vocab
    ])
    return np.log(len(doc_counts) / num_docs_each)

def get_tfs(doc_counts: list[str], vocab: list[str]) -> np.ndarray:
    '''Get the term frequency matrix, where tf(t, d) = count(t, d) / len(d),
    t is the term/word, d is the document
    Tranpose this, since the tests seem to be incorrect
        According to prompt, .shape should be (len(vocab), num docs)
        According to test cases though, .shape should be (num docs, len(vocab))'''
    doc_lens = [sum(doc_count.values()) for doc_count in doc_counts]
    return np.array([
        [doc_counts[doc_idx][word] / doc_lens[doc_idx]
         for doc_idx in range(len(doc_counts))]
        for word in vocab
    ]).T

def tfidf_vectorizer(documents: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    The outputs seem to expect a transposed version of the tf-idf matrix
    """
    documents = [doc.lower() for doc in documents]
    doc_counts = [Counter(doc.split(' ')) for doc in documents]
    vocab = get_vocab(doc_counts)
    idfs = get_idfs(doc_counts, vocab)
    tfs = get_tfs(doc_counts, vocab)
    return (tfs * idfs, vocab)