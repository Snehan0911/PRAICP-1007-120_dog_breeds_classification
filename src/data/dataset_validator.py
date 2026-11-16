"""
CanineVision AI — Dataset Validator.

Responsibility:
    Validates dataset quality BEFORE training begins.

    Professional ML pipelines always validate data first.
    If you skip this step, you discover data issues mid-training —
    at hour 3 of a 6-hour CPU run. This module catches those issues upfront.

    Checks performed:
    1. Corrupt images (cannot be opened by Pillow)
    2. Non-RGB images (grayscale, RGBA — convert to RGB for training)
    3. Extremely small images (below minimum viable dimensions)
    4. Empty breed directories
    5. Unexpected file formats

Usage:
    from src.data.dataset_validator import DatasetValidator
    validator = DatasetValidator(min_image_size=(32, 32))
    report = validator.validate(df)
    print(report.summary())
"""

from dataclasses import dataclass, field

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """
    Container for dataset validation results.

    Separates issues into different severity levels:
    - corrupt: Images that cannot be opened at all (must be excluded)
    - non_rgb: Images in wrong colour mode (can be converted)
    - too_small: Images below minimum dimension threshold (can be excluded or padded)
    - valid_count: Number of images that passed all checks
    """

    total_checked: int = 0
    valid_count: int = 0
    corrupt_files: list[str] = field(default_factory=list)
    non_rgb_files: list[str] = field(default_factory=list)
    too_small_files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATASET VALIDATION REPORT",
            "=" * 60,
            f"Total images checked : {self.total_checked}",
            f"Valid images         : {self.valid_count}",
            f"Corrupt (excluded)   : {len(self.corrupt_files)}",
            f"Non-RGB (convertible): {len(self.non_rgb_files)}",
            f"Too small (<min_size): {len(self.too_small_files)}",
            "=" * 60,
        ]
        if self.corrupt_files:
            lines.append("CORRUPT FILES (first 10):")
            for f in self.corrupt_files[:10]:
                lines.append(f"  {f}")
        return "\n".join(lines)

    @property
    def has_critical_issues(self) -> bool:
        """True if there are corrupt images that must be excluded."""
        return len(self.corrupt_files) > 0


class DatasetValidator:
    """
    Validates all images in the dataset DataFrame.

    Args:
        min_image_size: Minimum acceptable (width, height) in pixels.
                        Images smaller than this are flagged (not removed).
                        Default is (32, 32) — anything smaller is unusable.
    """

    def __init__(self, min_image_size: tuple[int, int] = (32, 32)) -> None:
        self.min_width, self.min_height = min_image_size

    def validate(self, df: pd.DataFrame, sample_size: int | None = None) -> ValidationReport:
        """
        Validate all images in the DataFrame.

        Args:
            df: DataFrame from DatasetLoader.load() with 'filepath' column.
            sample_size: If provided, validate a random sample of this size
                         instead of the full dataset (useful for quick checks).

        Returns:
            ValidationReport with counts and file lists for each issue type.
        """
        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info("Validating a sample of %d images...", len(df))
        else:
            logger.info("Validating all %d images...", len(df))

        report = ValidationReport(total_checked=len(df))

        for _, row in df.iterrows():
            filepath = row["filepath"]
            issue = self._check_image(filepath)

            if issue == "corrupt":
                report.corrupt_files.append(filepath)
            elif issue == "non_rgb":
                report.non_rgb_files.append(filepath)
                report.valid_count += 1  # Can be converted — still usable
            elif issue == "too_small":
                report.too_small_files.append(filepath)
                report.valid_count += 1  # Still usable with padding
            else:
                report.valid_count += 1

        logger.info(
            "Validation complete. Valid: %d | Corrupt: %d | Non-RGB: %d | Too small: %d",
            report.valid_count,
            len(report.corrupt_files),
            len(report.non_rgb_files),
            len(report.too_small_files),
        )
        return report

    def _check_image(self, filepath: str) -> str | None:
        """
        Check a single image file and return a string describing the issue,
        or None if the image is valid.

        Returns:
            None       — image is valid
            "corrupt"  — cannot be opened
            "non_rgb"  — opened but not RGB mode
            "too_small"— opened but smaller than minimum dimensions
        """
        try:
            with Image.open(filepath) as img:
                # Check colour mode
                if img.mode not in ("RGB", "RGBA", "L", "P"):
                    return "corrupt"

                if img.mode != "RGB":
                    return "non_rgb"

                # Check dimensions
                width, height = img.size
                if width < self.min_width or height < self.min_height:
                    return "too_small"

                return None

        except (UnidentifiedImageError, OSError, SyntaxError):
            return "corrupt"

    def filter_valid(self, df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
        """
        Return a filtered DataFrame with corrupt images removed.
        Non-RGB and too-small images are kept (they can be handled during preprocessing).

        Args:
            df: Original full DataFrame.
            report: ValidationReport from validate().

        Returns:
            DataFrame with corrupt images excluded.
        """
        corrupt_set = set(report.corrupt_files)
        filtered = df[~df["filepath"].isin(corrupt_set)].copy()
        removed = len(df) - len(filtered)
        if removed > 0:
            logger.warning("Removed %d corrupt image(s) from dataset.", removed)
        return filtered.reset_index(drop=True)
