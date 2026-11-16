"""
CanineVision AI — Comprehensive Evaluation Metrics.

Computes:
- Top-1 and Top-3 accuracy
- Macro and weighted Precision, Recall, F1-score
- Per-class metrics
- Inference latency statistics
"""

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def compute_top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    """
    Computes top-k categorical accuracy.

    Args:
        y_true: Ground truth integer class indices, shape (N,).
        y_prob: Predicted probability distributions, shape (N, num_classes).
        k: Rank threshold (e.g. 3).

    Returns:
        Top-k accuracy float between 0.0 and 1.0.
    """
    top_k_preds = np.argsort(y_prob, axis=-1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, None], axis=-1)
    return float(np.mean(correct))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_names: list[str] | None = None,
) -> dict:
    """
    Evaluates multi-class classification performance across all key metrics.

    Args:
        y_true: Ground truth integer labels (N,).
        y_pred: Predicted class indices (N,).
        y_prob: Predicted probability matrix (N, num_classes).
        target_names: Breed names ordered by class index.

    Returns:
        Dictionary of computed metric values.
    """
    metrics = {
        "top_1_accuracy": float(np.mean(y_true == y_pred)),
        "top_3_accuracy": compute_top_k_accuracy(y_true, y_prob, k=3),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["detailed_report"] = report

    return metrics
