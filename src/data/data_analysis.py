"""
CanineVision AI — Dataset Analysis (EDA).

Responsibility:
    Generates comprehensive dataset statistics and visualisations
    for the Exploratory Data Analysis (EDA) phase.

    EDA is NOT optional.
    Before building a model, you must understand your data:
    - Are some breeds severely under-represented?
    - What image dimensions does the model need to handle?
    - Are there corrupted or anomalous files?

    Every chart and statistic generated here feeds directly into:
    - Preprocessing decisions (image size, normalisation)
    - Training decisions (augmentation strength, class weights)
    - Model card documentation (dataset limitations)

Usage:
    from src.data.data_analysis import DatasetAnalyzer
    analyzer = DatasetAnalyzer(output_dir="outputs/eda")
    stats = analyzer.compute_class_distribution(df)
    analyzer.plot_class_distribution(stats)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Consistent styling for all plots
plt.style.use("seaborn-v0_8-whitegrid")
PLOT_DPI = 150
FIGURE_BG = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#4fc3f7"
PALETTE = "Blues_r"


class DatasetAnalyzer:
    """
    Generates EDA statistics and visualisations for the dog breed dataset.

    Args:
        output_dir: Directory where charts and reports are saved.
    """

    def __init__(self, output_dir: str | Path = "outputs/eda") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_class_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-class image counts and return a sorted DataFrame.

        Args:
            df: Full image DataFrame with 'label' column.

        Returns:
            DataFrame with columns: label, count, percentage
        """
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts["percentage"] = (counts["count"] / len(df) * 100).round(2)
        counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
        return counts

    def compute_image_dimension_stats(
        self, df: pd.DataFrame, sample_size: int = 500
    ) -> dict:
        """
        Sample images and compute width/height statistics.

        Args:
            df: DataFrame with 'filepath' column.
            sample_size: Number of images to sample for speed.

        Returns:
            Dictionary of dimension statistics.
        """
        sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        widths, heights, aspects = [], [], []

        for _, row in sample.iterrows():
            try:
                with Image.open(row["filepath"]) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspects.append(w / h)
            except Exception:
                pass

        stats = {
            "sample_size": len(widths),
            "width": {
                "min": int(np.min(widths)), "max": int(np.max(widths)),
                "mean": float(np.mean(widths)), "median": float(np.median(widths)),
            },
            "height": {
                "min": int(np.min(heights)), "max": int(np.max(heights)),
                "mean": float(np.mean(heights)), "median": float(np.median(heights)),
            },
            "aspect_ratio": {
                "min": round(float(np.min(aspects)), 3),
                "max": round(float(np.max(aspects)), 3),
                "mean": round(float(np.mean(aspects)), 3),
            },
        }
        return stats

    def compute_dataset_summary(self, df: pd.DataFrame) -> dict:
        """
        Compute high-level dataset statistics.

        Args:
            df: Full image DataFrame.

        Returns:
            Dictionary with key dataset facts.
        """
        dist = self.compute_class_distribution(df)
        return {
            "total_images": int(len(df)),
            "num_classes": int(df["label"].nunique()),
            "images_per_class": {
                "min": int(dist["count"].min()),
                "max": int(dist["count"].max()),
                "mean": round(float(dist["count"].mean()), 1),
                "std": round(float(dist["count"].std()), 1),
            },
            "class_with_fewest": dist.iloc[-1]["label"],
            "class_with_most": dist.iloc[0]["label"],
            "imbalance_ratio": round(
                float(dist["count"].max() / dist["count"].min()), 2
            ),
        }

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_class_distribution(
        self, dist_df: pd.DataFrame, top_n: int = 30
    ) -> Path:
        """
        Plot a horizontal bar chart showing image counts per breed.

        Args:
            dist_df: DataFrame from compute_class_distribution().
            top_n: Show only the top N and bottom N breeds for readability.

        Returns:
            Path to saved chart.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.patch.set_facecolor(FIGURE_BG)

        for ax in axes:
            ax.set_facecolor(FIGURE_BG)

        # Top N
        top = dist_df.head(top_n)
        axes[0].barh(top["label"], top["count"], color=ACCENT_COLOR, alpha=0.85)
        axes[0].set_xlabel("Image Count", color=TEXT_COLOR, fontsize=11)
        axes[0].set_title(f"Top {top_n} Breeds by Image Count", color=TEXT_COLOR, fontsize=13, pad=12)
        axes[0].tick_params(colors=TEXT_COLOR)
        axes[0].invert_yaxis()

        # Bottom N
        bottom = dist_df.tail(top_n).sort_values("count", ascending=True)
        axes[1].barh(bottom["label"], bottom["count"], color="#ff7043", alpha=0.85)
        axes[1].set_xlabel("Image Count", color=TEXT_COLOR, fontsize=11)
        axes[1].set_title(f"Bottom {top_n} Breeds by Image Count", color=TEXT_COLOR, fontsize=13, pad=12)
        axes[1].tick_params(colors=TEXT_COLOR)

        for ax in axes:
            ax.spines[:].set_color("#444")
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle("CanineVision AI — Class Distribution (120 Breeds)",
                     color=TEXT_COLOR, fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        output_path = self.output_dir / "class_distribution.png"
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=FIGURE_BG)
        plt.close(fig)
        logger.info("Saved class distribution chart: %s", output_path)
        return output_path

    def plot_imbalance_histogram(self, dist_df: pd.DataFrame) -> Path:
        """
        Plot a histogram showing the distribution of images-per-class counts.

        Args:
            dist_df: DataFrame from compute_class_distribution().

        Returns:
            Path to saved chart.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(FIGURE_BG)
        ax.set_facecolor(FIGURE_BG)

        ax.hist(dist_df["count"], bins=25, color=ACCENT_COLOR, alpha=0.8, edgecolor="#333")
        ax.axvline(dist_df["count"].mean(), color="#ff7043", linestyle="--",
                   linewidth=1.5, label=f"Mean: {dist_df['count'].mean():.0f}")
        ax.set_xlabel("Images per Class", color=TEXT_COLOR)
        ax.set_ylabel("Number of Classes", color=TEXT_COLOR)
        ax.set_title("Class Imbalance Distribution", color=TEXT_COLOR, fontsize=13)
        ax.tick_params(colors=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, facecolor="#2a2a3e")

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / "imbalance_histogram.png"
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=FIGURE_BG)
        plt.close(fig)
        logger.info("Saved imbalance histogram: %s", output_path)
        return output_path

    def save_summary_json(self, summary: dict) -> Path:
        """
        Save the dataset summary dictionary as a JSON file.

        Args:
            summary: Dictionary from compute_dataset_summary().

        Returns:
            Path to saved JSON file.
        """
        output_path = self.output_dir / "dataset_summary.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Dataset summary saved: %s", output_path)
        return output_path
