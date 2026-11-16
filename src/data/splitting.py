"""
CanineVision AI — Dataset Splitting.

Responsibility:
    Splits the full image DataFrame into train, validation, and test sets
    using a stratified approach.

    WHY STRATIFIED?
    Without stratification, a random split might put all 148 images of
    the 'redbone' breed into training, leaving zero for validation.
    Stratified splitting ensures EVERY breed has proportional representation
    in all three sets.

    WHY THREE SETS?
    - Training set: the model learns from this
    - Validation set: used DURING training to monitor overfitting (the model
                      never back-propagates through this data)
    - Test set: NEVER touched until final evaluation — gives you an unbiased
                measure of real-world performance

    WHY NO LEAKAGE?
    We split BEFORE any augmentation or preprocessing. If you augmented first
    and then split, augmented versions of a training image could appear in
    the test set — that would artificially inflate test accuracy.

Usage:
    from src.data.splitting import DatasetSplitter
    splitter = DatasetSplitter(train=0.70, val=0.15, test=0.15)
    train_df, val_df, test_df = splitter.split(df)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Stratified train/validation/test splitter.

    Args:
        train: Fraction of data for training (e.g., 0.70).
        val:   Fraction of data for validation (e.g., 0.15).
        test:  Fraction of data for testing (e.g., 0.15).
        seed:  Random seed for reproducibility.
    """

    def __init__(
        self,
        train: float = 0.70,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
    ) -> None:
        total = round(train + val + test, 10)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Train + val + test must sum to 1.0, got {total:.4f}"
            )

        self.train_frac = train
        self.val_frac = val
        self.test_frac = test
        self.seed = seed

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into train, validation, and test sets.

        Process:
            1. Split full dataset into (train+val) and test using stratification.
            2. Split (train+val) into train and val using stratification.

        Args:
            df: Full image DataFrame from DatasetLoader with 'label' column.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            ValueError: If any class has fewer than 3 images
                        (not enough for stratified split).
        """
        self._check_min_samples(df)

        logger.info(
            "Splitting %d images → train=%.0f%% / val=%.0f%% / test=%.0f%%",
            len(df),
            self.train_frac * 100,
            self.val_frac * 100,
            self.test_frac * 100,
        )

        # Step 1: Separate test set from the rest
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_frac,
            stratify=df["label"],
            random_state=self.seed,
        )

        # Step 2: Separate validation from training
        # val_frac relative to the remaining (train_val) portion:
        relative_val_frac = self.val_frac / (self.train_frac + self.val_frac)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_frac,
            stratify=train_val_df["label"],
            random_state=self.seed,
        )

        # Add a 'split' column for transparency
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        logger.info(
            "Split complete → train: %d | val: %d | test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        self._verify_no_leakage(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        """
        Save split DataFrames to CSV files in the output directory.

        Args:
            train_df: Training split DataFrame.
            val_df:   Validation split DataFrame.
            test_df:  Test split DataFrame.
            output_dir: Directory to write CSV files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        logger.info("Split CSVs saved to: %s", output_dir)

    @staticmethod
    def _check_min_samples(df: pd.DataFrame, min_samples: int = 3) -> None:
        """
        Ensure every class has enough samples for stratified splitting.
        Stratification needs at least as many samples as the number of splits.
        """
        counts = df["label"].value_counts()
        problem_classes = counts[counts < min_samples]
        if not problem_classes.empty:
            raise ValueError(
                f"{len(problem_classes)} class(es) have fewer than {min_samples} samples "
                f"and cannot be stratified:\n{problem_classes}"
            )

    @staticmethod
    def _verify_no_leakage(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """
        Assert that no image appears in more than one split.
        This is a data integrity check — it should never fail, but
        we verify explicitly to catch any future bugs.
        """
        train_paths = set(train_df["filepath"])
        val_paths = set(val_df["filepath"])
        test_paths = set(test_df["filepath"])

        train_val_overlap = train_paths & val_paths
        train_test_overlap = train_paths & test_paths
        val_test_overlap = val_paths & test_paths

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise AssertionError(
                f"DATA LEAKAGE DETECTED!\n"
                f"  Train∩Val: {len(train_val_overlap)} duplicates\n"
                f"  Train∩Test: {len(train_test_overlap)} duplicates\n"
                f"  Val∩Test: {len(val_test_overlap)} duplicates"
            )

        logger.info("Data leakage check passed — zero overlaps between splits.")
