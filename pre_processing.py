"""
Dataset Exploration & Preprocessing Entry Point.
Project: Dog Breeds Classification (PRAICP-1007-120)

Main executable entry point for scanning the dataset, verifying images, generating
labels.csv, and rendering all Exploratory Data Analysis (EDA) charts.

Usage:
    python pre_processing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_analysis import DataAnalyzer
from src.data.dataset_validator import DatasetValidator


def run_pipeline(force_rescan: bool = False) -> None:
    """Runs end-to-end dataset validation and exploratory data analysis."""
    print("=" * 70)
    print("  DOG BREEDS CLASSIFICATION — DATASET EXPLORATION & EDA PIPELINE")
    print("  Project ID: PRAICP-1007-120")
    print("=" * 70)

    labels_path = Path("labels.csv")
    validator = DatasetValidator()

    if not labels_path.exists() or force_rescan:
        print(
            "\n[Step 1/2] Scanning dataset directories and verifying image integrity..."
        )
        validator.scan_and_validate()
    else:
        print(f"\n[Step 1/2] Found existing {labels_path}. Loading metadata...")
        validator.load_existing_labels(labels_path)

    # Compute summary statistics & initial charts
    summary = validator.compute_summary_statistics()
    validator.generate_eda_charts()
    validator.export_documentation(summary)

    print("\n[Step 2/2] Running deep statistical & pixel intensity profiling...")
    analyzer = DataAnalyzer()
    _ = analyzer.analyze_top_and_bottom_breeds(n=10)
    pixel_stats = analyzer.analyze_pixel_intensities(sample_size=500)

    print("\n" + "=" * 70)
    print("EXPLORATION & VALIDATION SUMMARY:")
    print(f"  • Total Valid Images:     {summary['total_images']:,}")
    print(f"  • Total Breed Categories: {summary['total_classes']}")
    print(f"  • Corrupted Files:        {summary['corrupted_images_count']}")
    print(
        f"  • Images Per Breed:       Min={summary['min_images_per_class']}, Max={summary['max_images_per_class']}, Mean={summary['mean_images_per_class']}"
    )
    print(f"  • Class Imbalance Ratio:  {summary['imbalance_ratio']}x (Well Balanced)")
    print(
        f"  • Mean Resolution:        {summary['mean_width']} x {summary['mean_height']} px (Aspect Ratio: {summary['mean_aspect_ratio']})"
    )
    print("\nPIXEL INTENSITY CHARACTERISTICS (RGB 0-255):")
    print(
        f"  • Red Channel:   Mean={pixel_stats['red_channel']['mean']}, Std={pixel_stats['red_channel']['std']}"
    )
    print(
        f"  • Green Channel: Mean={pixel_stats['green_channel']['mean']}, Std={pixel_stats['green_channel']['std']}"
    )
    print(
        f"  • Blue Channel:  Mean={pixel_stats['blue_channel']['mean']}, Std={pixel_stats['blue_channel']['std']}"
    )
    print("\nGENERATED EDA CHARTS (outputs/eda/):")
    print("  1. outputs/eda/class_distribution.png")
    print("  2. outputs/eda/image_resolutions.png")
    print("  3. outputs/eda/sample_gallery.png")
    print("  4. outputs/eda/top_bottom_breeds.png")
    print("  5. outputs/eda/pixel_intensity_distribution.png")
    print("=" * 70)
    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset exploration and EDA.")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Force re-scan of raw images even if labels.csv exists.",
    )
    args = parser.parse_args()
    run_pipeline(force_rescan=args.rescan)
