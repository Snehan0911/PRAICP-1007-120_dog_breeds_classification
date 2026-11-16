"""
CanineVision AI — Dataset inspection CLI script.

This script performs a full dataset inspection and prints a report.
Run from the project root:

    python scripts/inspect_dataset.py

No arguments required. All paths are discovered automatically.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_analysis import DatasetAnalyzer
from src.data.dataset_loader import DatasetLoader
from src.data.dataset_validator import DatasetValidator
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)


def main() -> None:
    """Run a full dataset inspection and print a report to console."""

    set_all_seeds(42)
    paths = ProjectPaths()
    paths.ensure_all_exist()

    print("\n" + "=" * 70)
    print("  CanineVision AI — Dataset Inspection Report")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Step 1: Load dataset
    # -----------------------------------------------------------------
    print("\n[1/4] Loading dataset...")
    loader = DatasetLoader(dataset_root=paths.dataset_root)
    df = loader.load()
    print(f"      Total images discovered : {len(df):,}")
    print(f"      Total breed classes     : {df['label'].nunique()}")

    # -----------------------------------------------------------------
    # Step 2: Validate dataset (sample for speed)
    # -----------------------------------------------------------------
    print("\n[2/4] Validating images (500-image sample)...")
    validator = DatasetValidator(min_image_size=(32, 32))
    report = validator.validate(df, sample_size=500)
    print(report.summary())

    # -----------------------------------------------------------------
    # Step 3: Compute statistics
    # -----------------------------------------------------------------
    print("\n[3/4] Computing dataset statistics...")
    analyzer = DatasetAnalyzer(output_dir=paths.outputs_eda)

    dist_df = analyzer.compute_class_distribution(df)
    summary = analyzer.compute_dataset_summary(df)
    dim_stats = analyzer.compute_image_dimension_stats(df, sample_size=300)

    print("\n  DATASET SUMMARY:")
    for key, value in summary.items():
        print(f"    {key:<30} {value}")

    print("\n  IMAGE DIMENSION STATS (sample of 300):")
    print(f"    Width  — min: {dim_stats['width']['min']}  max: {dim_stats['width']['max']}  mean: {dim_stats['width']['mean']:.0f}")
    print(f"    Height — min: {dim_stats['height']['min']}  max: {dim_stats['height']['max']}  mean: {dim_stats['height']['mean']:.0f}")
    print(f"    Aspect — min: {dim_stats['aspect_ratio']['min']}  max: {dim_stats['aspect_ratio']['max']}  mean: {dim_stats['aspect_ratio']['mean']}")

    print("\n  TOP 5 BREEDS BY IMAGE COUNT:")
    for _, row in dist_df.head(5).iterrows():
        print(f"    {row['label']:<40} {row['count']} images ({row['percentage']:.2f}%)")

    print("\n  BOTTOM 5 BREEDS BY IMAGE COUNT:")
    for _, row in dist_df.tail(5).iterrows():
        print(f"    {row['label']:<40} {row['count']} images ({row['percentage']:.2f}%)")

    # -----------------------------------------------------------------
    # Step 4: Save reports and charts
    # -----------------------------------------------------------------
    print("\n[4/4] Saving charts and reports...")

    # Save distribution chart
    chart_path = analyzer.plot_class_distribution(dist_df)
    print(f"      Saved: {chart_path}")

    # Save histogram
    hist_path = analyzer.plot_imbalance_histogram(dist_df)
    print(f"      Saved: {hist_path}")

    # Save summary JSON
    summary["image_dimension_stats"] = dim_stats
    json_path = analyzer.save_summary_json(summary)
    print(f"      Saved: {json_path}")

    # Save full distribution CSV
    csv_path = paths.outputs_eda / "class_distribution.csv"
    dist_df.to_csv(csv_path, index=False)
    print(f"      Saved: {csv_path}")

    print("\n" + "=" * 70)
    print("  Dataset inspection complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
