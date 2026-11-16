"""
CanineVision AI — Confusion Matrix Visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_top_confused_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_n: int = 15,
    output_path: str | Path | None = None,
) -> Path | None:
    """
    Since 120x120 is unreadable in a standard figure, this plots the sub-matrix
    of the top N classes with the highest error rates.

    Args:
        y_true: Ground truth class integers.
        y_pred: Predicted class integers.
        class_names: Breed names in index order.
        top_n: Number of most error-prone classes to visualize.
        output_path: Destination image path.

    Returns:
        Path to saved plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class error count (off-diagonal elements)
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    top_error_indices = np.argsort(errors_per_class)[::-1][:top_n]

    sub_cm = cm[np.ix_(top_error_indices, top_error_indices)]
    sub_names = [class_names[i] for i in top_error_indices]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sub_names,
        yticklabels=sub_names,
        cbar=True,
        ax=ax,
    )

    ax.set_title(f"Confusion Sub-Matrix: Top {top_n} Most Confused Breeds", color="#ffffff", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Breed", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("True Breed", color="#e0e0e0", fontsize=11)
    ax.tick_params(colors="#e0e0e0", rotation=45)

    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        logger.info("Saved confusion sub-matrix to: %s", p)
        return p

    plt.close(fig)
    return None
