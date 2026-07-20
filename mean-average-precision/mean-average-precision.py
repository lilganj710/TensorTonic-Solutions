import numpy as np


def mean_average_precision(
        y_true_list: list[list[float]], y_score_list: list[list[float]],
        cutoff_query_len: int | None=None) -> tuple[float, np.ndarray]:
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
        Based on the example in the graphic, I don't think we can assume
        that the 2d lists are rectangular
    For each query (in y_true_list), the item is either relevant or irrelevant (1 or 0)
        As implied, this is what's used to compute precision
    Where are the scores used then? They're not directly in the objective function
        It seems that they're used to sort the queries in each list in y_true_list

    Optional: cutoff_query_len clips the max length of each query
        Possible bug: according to test caess, cutoff_query_len should be applied
        AFTER counting total relevant items
    """
    query_sort_idxs = [
        np.flip(np.argsort(y_score_list[query_idx]))
        for query_idx in range(len(y_true_list))
    ]
    sorted_query_list = [
        np.array(query)[query_sort_idxs]
        for query, query_sort_idxs in
        zip(y_true_list, query_sort_idxs, strict=True)
    ]
    average_precisions: list[float] = []
    for query_relevances in sorted_query_list:
        total_relevant_items = np.sum(query_relevances)
        if total_relevant_items == 0:
            average_precisions.append(0)
            continue
        query_relevances = query_relevances[:cutoff_query_len]
        query_precisions = np.cumsum(query_relevances) / np.arange(1, len(query_relevances)+1)
        average_precisions.append(
            np.sum(query_precisions * query_relevances) / total_relevant_items)
    return np.mean(average_precisions), np.array(average_precisions)