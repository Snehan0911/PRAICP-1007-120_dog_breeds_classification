"""
Exploratory Data Analysis (EDA) Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Performs pixel intensity profiling, RGB channel distributions, aspect ratio analysis,
and generates publication-quality visual diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DataAnalysis")


class DataAnalyzer:
    """
    Performs comprehensive Exploratory Data Analysis on the 120 Dog Breeds dataset.
    """

    def __init__(
        self,
        labels_csv_path: str | Path = "labels.csv",
        eda_output_dir: str | Path = "outputs/eda",
    ) -> None:
        self.labels_csv_path = Path(labels_csv_path)
        self.eda_output_dir = Path(eda_output_dir)
        self.eda_output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_labels()

    def _load_labels(self) -> pd.DataFrame:
        """Loads and validates labels.csv."""
        if not self.labels_csv_path.exists():
            raise FileNotFoundError(
                f"Missing {self.labels_csv_path}. Please run dataset_validator.py first."
            )
        df = pd.read_csv(self.labels_csv_path)
        logger.info(f"Loaded {len(df)} records from {self.labels_csv_path}")
        return df

    def analyze_top_and_bottom_breeds(self, n: int = 10) -> dict[str, Any]:
        """Identifies top-n and bottom-n breeds by image count and plots comparisons."""
        class_counts = self.df["breed"].value_counts()
        top_n = class_counts.head(n)
        bottom_n = class_counts.tail(n).iloc[::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        # Top N
        y_pos1 = np.arange(len(top_n))
        ax1.barh(y_pos1, top_n.values, color="#2b5c8f", edgecolor="black")
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(top_n.index, fontsize=11)
        ax1.invert_yaxis()
        ax1.set_xlabel("Number of Images", fontsize=12)
        ax1.set_title(
            f"Top {n} Breeds with Highest Image Counts", fontsize=14, fontweight="bold"
        )
        for i, v in enumerate(top_n.values):
            ax1.text(v + 1, i, f" {v}", va="center", fontsize=10, fontweight="bold")

        # Bottom N
        y_pos2 = np.arange(len(bottom_n))
        ax2.barh(y_pos2, bottom_n.values, color="#c0392b", edgecolor="black")
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(bottom_n.index, fontsize=11)
        ax2.invert_yaxis()
        ax2.set_xlabel("Number of Images", fontsize=12)
        ax2.set_title(
            f"Bottom {n} Breeds with Lowest Image Counts",
            fontsize=14,
            fontweight="bold",
        )
        for i, v in enumerate(bottom_n.values):
            ax2.text(v + 1, i, f" {v}", va="center", fontsize=10, fontweight="bold")

        plt.suptitle(
            "Dog Breeds Classification — Top vs. Bottom Breed Representation",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        out_path = self.eda_output_dir / "top_bottom_breeds.png"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved top/bottom breeds chart to {out_path}")

        return {
            "top_n": top_n.to_dict(),
            "bottom_n": bottom_n.to_dict(),
            "chart_path": str(out_path),
        }

    def analyze_pixel_intensities(self, sample_size: int = 500) -> dict[str, Any]:
        """
        Samples images across breeds to compute RGB channel means, standard deviations,
        and pixel intensity distributions.
        """
        logger.info(
            f"Sampling {sample_size} images for RGB channel and pixel intensity profiling..."
        )
        sampled_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)

        r_vals: list[float] = []
        g_vals: list[float] = []
        b_vals: list[float] = []

        for _, row in sampled_df.iterrows():
            img_path = Path(row["filepath"])
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    # Downsample slightly for fast memory-safe histogram extraction
                    small = img_rgb.resize((64, 64))
                    arr = np.array(small, dtype=np.float32)
                    r_vals.extend(arr[:, :, 0].flatten().tolist()[::4])
                    g_vals.extend(arr[:, :, 1].flatten().tolist()[::4])
                    b_vals.extend(arr[:, :, 2].flatten().tolist()[::4])
            except (OSError, ValueError):
                continue

        r_arr = np.array(r_vals)
        g_arr = np.array(g_vals)
        b_arr = np.array(b_vals)

        r_mean, r_std = float(r_arr.mean()), float(r_arr.std())
        g_mean, g_std = float(g_arr.mean()), float(g_arr.std())
        b_mean, b_std = float(b_arr.mean()), float(b_arr.std())

        # Generate Pixel Intensity Histogram Plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.hist(
            r_arr,
            bins=64,
            range=(0, 255),
            color="red",
            alpha=0.45,
            label=f"Red (Mean: {r_mean:.1f}, Std: {r_std:.1f})",
            density=True,
        )
        ax.hist(
            g_arr,
            bins=64,
            range=(0, 255),
            color="green",
            alpha=0.45,
            label=f"Green (Mean: {g_mean:.1f}, Std: {g_std:.1f})",
            density=True,
        )
        ax.hist(
            b_arr,
            bins=64,
            range=(0, 255),
            color="blue",
            alpha=0.45,
            label=f"Blue (Mean: {b_mean:.1f}, Std: {b_std:.1f})",
            density=True,
        )

        ax.set_title(
            "Dog Breeds Classification — Global Pixel Intensity Distribution by Channel",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Pixel Value (0 - 255)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_xlim(0, 255)
        ax.legend(fontsize=11)
        plt.tight_layout()

        out_path = self.eda_output_dir / "pixel_intensity_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved pixel intensity chart to {out_path}")

        return {
            "red_channel": {"mean": round(r_mean, 2), "std": round(r_std, 2)},
            "green_channel": {"mean": round(g_mean, 2), "std": round(g_std, 2)},
            "blue_channel": {"mean": round(b_mean, 2), "std": round(b_std, 2)},
            "chart_path": str(out_path),
        }


if __name__ == "__main__":
    analyzer = DataAnalyzer()
    breed_stats = analyzer.analyze_top_and_bottom_breeds()
    intensity_stats = analyzer.analyze_pixel_intensities()
    print("EDA Complete:")
    print("  Intensity stats:", intensity_stats)
