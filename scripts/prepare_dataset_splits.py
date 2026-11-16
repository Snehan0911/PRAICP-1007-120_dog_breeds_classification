"""
CanineVision AI — Dataset Preparation and Splitting Script.

This script:
1. Loads all 20,580 images across 120 breeds.
2. Performs stratified splitting (70% train, 15% val, 15% test).
3. Verifies zero data leakage across splits.
4. Saves train.csv, val.csv, test.csv to data/splits/.
5. Saves the canonical class_indices.json to artifacts/class_mapping/.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.data.splitting import DatasetSplitter
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)


def main() -> None:
    set_all_seeds(42)
    paths = ProjectPaths()
    paths.ensure_all_exist()

    logger.info("Starting dataset splitting pipeline...")

    # 1. Discover all images
    loader = DatasetLoader(dataset_root=paths.dataset_root)
    df = loader.load()

    # 2. Extract and save class index mapping
    class_mapping = loader.get_class_mapping()
    mapping_file = paths.artifacts_class_mapping / "class_indices.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    logger.info("Saved class mapping (120 classes) to: %s", mapping_file)

    # 3. Create stratified splits (70% train, 15% val, 15% test)
    splitter = DatasetSplitter(train=0.70, val=0.15, test=0.15, seed=42)
    train_df, val_df, test_df = splitter.split(df)

    # 4. Save CSV splits
    splitter.save_splits(train_df, val_df, test_df, output_dir=paths.data_splits)

    print("\n" + "=" * 60)
    print("  DATASET SPLITTING SUMMARY")
    print("=" * 60)
    print(f"  Total Images    : {len(df):,}")
    print(f"  Training Set    : {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation Set  : {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test Set        : {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Number of Breeds: {len(class_mapping)}")
    print(f"  Class Mapping   : {mapping_file}")
    print(f"  Splits Saved To : {paths.data_splits}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
