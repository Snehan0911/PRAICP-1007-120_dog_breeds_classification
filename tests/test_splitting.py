"""
Tests — Dataset splitting and data leakage prevention.
"""

import pytest


def test_splitting_ratios_and_no_leakage(small_dataset_df):
    """Verify that splitting produces exact requested proportions with zero overlap."""
    from src.data.splitting import DatasetSplitter

    splitter = DatasetSplitter(train=0.60, val=0.20, test=0.20, seed=42)
    train_df, val_df, test_df = splitter.split(small_dataset_df)

    assert len(train_df) + len(val_df) + len(test_df) == len(small_dataset_df)

    # Check zero overlap
    train_set = set(train_df["filepath"])
    val_set = set(val_df["filepath"])
    test_set = set(test_df["filepath"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_invalid_ratios_raise_error():
    """Ratios that don't sum to 1.0 must raise ValueError."""
    from src.data.splitting import DatasetSplitter

    with pytest.raises(ValueError, match="must sum to 1.0"):
        DatasetSplitter(train=0.70, val=0.20, test=0.20)
