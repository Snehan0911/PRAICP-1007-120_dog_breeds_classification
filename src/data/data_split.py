"""
Stratified Dataset Splitting Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Performs stratified train/validation/test splitting (70% / 15% / 15%)
ensuring zero data leakage and identical class balance across all subsets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DataSplitter")


def create_stratified_splits(
    labels_csv_path: str | Path = "labels.csv",
    output_dir: str | Path = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits labels.csv into stratified train, val, and test partitions, saves them as CSVs,
    and returns the three DataFrames.

    Args:
        labels_csv_path: Path to master labels.csv.
        output_dir: Directory where train.csv, val.csv, and test.csv will be saved.
        train_ratio: Proportion for training (default 0.70).
        val_ratio: Proportion for validation (default 0.15).
        test_ratio: Proportion for testing (default 0.15).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    total = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0. Got {total:.4f}")

    labels_path = Path(labels_csv_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at: {labels_path}")

    df = pd.read_csv(labels_path)
    logger.info(f"Loaded {len(df)} records from {labels_path}")

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=temp_ratio, random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(df, df["breed"]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    # Second split: Val vs Test from Temp
    val_rel_ratio = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - val_rel_ratio), random_state=random_state
    )
    val_idx, test_idx = next(sss2.split(temp_df, temp_df["breed"]))

    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    # Verify zero data leakage
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    assert len(train_ids.intersection(val_ids)) == 0, (
        "Data leakage detected: Train and Val overlap!"
    )
    assert len(train_ids.intersection(test_ids)) == 0, (
        "Data leakage detected: Train and Test overlap!"
    )
    assert len(val_ids.intersection(test_ids)) == 0, (
        "Data leakage detected: Val and Test overlap!"
    )
    assert len(train_df) + len(val_df) + len(test_df) == len(df), (
        "Total record count mismatch!"
    )

    # Verify all 120 classes are present in each split
    assert train_df["breed"].nunique() == 120, "Missing classes in train split!"
    assert val_df["breed"].nunique() == 120, "Missing classes in val split!"
    assert test_df["breed"].nunique() == 120, "Missing classes in test split!"

    # Save to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Stratified splits exported to {out_dir}: "
        f"Train={len(train_df)} ({len(train_df) / len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df) / len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df) / len(df):.1%})"
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    train, val, test = create_stratified_splits()
    print("Stratified Split Complete:")
    print(f"  Train: {len(train):,} samples, {train['breed'].nunique()} classes")
    print(f"  Val:   {len(val):,} samples, {val['breed'].nunique()} classes")
    print(f"  Test:  {len(test):,} samples, {test['breed'].nunique()} classes")
