"""
Confusion Matrix Generation and Visualization Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Computes, normalizes, and plots publication-grade 120-class confusion matrix heatmaps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ConfusionMatrix")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: str | Path = "outputs/evaluation",
) -> tuple[Path, Path]:
    """
    Computes and plots both raw count and normalized confusion matrices.

    Args:
        y_true: Ground truth integer class array.
        y_pred: Predicted integer class array.
        class_names: List of all 120 breed names.
        output_dir: Directory where PNG plots will be saved.

    Returns:
        tuple[Path, Path]: (raw_matrix_path, normalized_matrix_path)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Normalized confusion matrix (per-class recall percentages)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1, keepdims=True))

    # 1. Full 120x120 Heatmap
    fig, ax = plt.subplots(figsize=(18, 16), dpi=150)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        "Dog Breeds Classification — Normalized Confusion Matrix (120 Breeds)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Class Index (0 - 119)", fontsize=13, labelpad=10)
    ax.set_ylabel("True Class Index (0 - 119)", fontsize=13, labelpad=10)
    plt.tight_layout()

    norm_path = out_path / "confusion_matrix_normalized.png"
    fig.savefig(norm_path)
    plt.close(fig)
    logger.info(f"Saved normalized confusion matrix to {norm_path}")

    # 2. Focused 20-Breed Submatrix Heatmap (with breed names for presentation)
    top_indices = np.argsort(np.bincount(y_true))[-20:]
    sub_cm = cm_norm[np.ix_(top_indices, top_indices)]
    sub_names = [class_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    im = ax.imshow(sub_cm, interpolation="nearest", cmap="YlGnBu")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(sub_names)))
    ax.set_yticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(sub_names, fontsize=9)

    ax.set_title(
        "Dog Breeds Classification — Top 20 Classes Confusion Sub-Matrix",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted Breed", fontsize=12)
    ax.set_ylabel("True Breed", fontsize=12)
    plt.tight_layout()

    focused_path = out_path / "confusion_matrix_top20.png"
    fig.savefig(focused_path)
    plt.close(fig)
    logger.info(f"Saved focused 20-breed confusion matrix to {focused_path}")

    return norm_path, focused_path
