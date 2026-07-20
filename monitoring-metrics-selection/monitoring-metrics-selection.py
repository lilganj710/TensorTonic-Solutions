import numpy as np
from numpy.typing import ArrayLike


def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    '''Return the classification metrics as described,
    keyed by names (accuracy, precision, recall, f1)'''
    confusion_matrix_entries = np.bincount(2*y_true + y_pred, minlength=4)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix_entries
    accuracy = (true_positives + true_negatives) / len(y_true)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0 else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0 else 0
    )
    f1_score = (
        2*precision*recall / (precision + recall)
        if precision + recall > 0 else 0
    )
    return {
        'accuracy': accuracy,
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
    }

def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    '''Return the regression metrics keyed by name: (mae, rmse)'''
    return {
        'mae': np.mean(np.abs(y_true - y_pred)),
        'rmse': np.sqrt(np.mean((y_true - y_pred)**2))
    }

def get_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    '''Return the ranking metrics keyed by name: (precision_at_3, recall_at_3)'''
    score_argsort = np.argsort(y_pred)
    top_idxs = score_argsort[-3:]
    total_relevant = sum(y_true)
    recall = (
        sum(y_true[top_idxs]) / total_relevant
        if total_relevant > 0 else 0
    )
    return {
        'precision_at_3': sum(y_true[top_idxs]) / 3,
        'recall_at_3': recall
    }

def compute_monitoring_metrics(system_type: str, y_true: ArrayLike, y_pred: ArrayLike):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    metric_functions = {
        'classification': get_classification_metrics,
        'regression': get_regression_metrics,
        'ranking': get_ranking_metrics
    }
    if system_type not in metric_functions:
        raise ValueError(f'{system_type=} invalid')
    metrics = metric_functions[system_type](np.array(y_true), np.array(y_pred))
    return list(metrics.items())