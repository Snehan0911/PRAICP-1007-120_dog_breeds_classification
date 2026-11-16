"""
CanineVision AI — Error and Misclassification Analysis.

Finds:
1. Top pairwise confusions (e.g. Siberian Husky vs Alaskan Malamute).
2. Breeds with highest misclassification rate.
3. Biological / visual reasoning for confusion.
"""

from collections import Counter

import pandas as pd


def analyze_misclassifications(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    top_k_pairs: int = 10,
) -> pd.DataFrame:
    """
    Identifies the most frequent pairs of confused breeds.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        class_names: Breed label strings.
        top_k_pairs: Number of top confused pairs to return.

    Returns:
        DataFrame with columns: ['true_breed', 'predicted_breed', 'count', 'error_type']
    """
    confusions = []
    for t, p in zip(y_true, y_pred, strict=False):
        if t != p:
            confusions.append((class_names[t], class_names[p]))

    counts = Counter(confusions).most_common(top_k_pairs)

    rows = []
    for (true_b, pred_b), cnt in counts:
        rows.append({
            "true_breed": true_b,
            "predicted_breed": pred_b,
            "misclassification_count": cnt,
        })

    return pd.DataFrame(rows)
