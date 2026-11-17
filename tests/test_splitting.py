"""
Unit tests for stratified dataset splitting and leakage prevention.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.data_split import create_stratified_splits


def test_stratified_splits_ratios_and_no_leakage():
    splits_dir = Path("data/splits")
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    assert train_path.exists(), "train.csv does not exist!"
    assert val_path.exists(), "val.csv does not exist!"
    assert test_path.exists(), "test.csv does not exist!"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    total_len = len(train_df) + len(val_df) + len(test_df)
    assert total_len == 20580

    # Test proportions (~70%, ~15%, ~15%)
    assert len(train_df) == pytest.approx(20580 * 0.70, abs=10)
    assert len(val_df) == pytest.approx(20580 * 0.15, abs=10)
    assert len(test_df) == pytest.approx(20580 * 0.15, abs=10)

    # Test zero data leakage
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share IDs!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share IDs!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share IDs!"

    # Test all 120 classes present in every split
    assert train_df["breed"].nunique() == 120
    assert val_df["breed"].nunique() == 120
    assert test_df["breed"].nunique() == 120


def test_invalid_split_ratios_raise_error():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        create_stratified_splits(train_ratio=0.80, val_ratio=0.10, test_ratio=0.05)
