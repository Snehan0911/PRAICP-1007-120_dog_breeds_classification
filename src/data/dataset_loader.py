"""
CanineVision AI — Dataset Loader.

Responsibility:
    Discovers all valid image files from the 120-class breed dataset
    and builds a structured DataFrame with file paths and labels.

    This module is intentionally separate from preprocessing:
    - Loading = discovering files and building a file list
    - Preprocessing = reading pixels and transforming them

    Keeping them separate means you can inspect the file list,
    check for issues, and generate splits BEFORE touching image pixels.

Usage:
    from src.data.dataset_loader import DatasetLoader
    loader = DatasetLoader(dataset_root="dog-breed-120/Dogs class")
    df = loader.load()
    print(df.head())
    # filepath | label | breed_name | class_index
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported image file extensions (all images in this dataset are .jpg,
# but we support others for future-proofing)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Folders inside the dataset directory that are NOT breed classes
JUNK_FOLDERS = {"Untitled Folder"}


class DatasetLoader:
    """
    Discovers and catalogues all images in the dog breed dataset.

    The dataset follows ImageNet naming convention:
        nXXXXXXXX-BreedName/
            image_001.jpg
            image_002.jpg
            ...

    The class label is extracted from the folder name by stripping
    the numeric synset prefix (e.g., 'n02085620-Chihuahua' → 'Chihuahua').

    Attributes:
        dataset_root: Path to the root containing 120 breed subdirectories.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root.resolve()}\n"
                "Ensure the dataset has been extracted correctly."
            )

    def load(self) -> pd.DataFrame:
        """
        Walk through all breed directories and collect image file paths.

        Returns:
            A pandas DataFrame with columns:
            - filepath (str): Absolute path to the image file
            - label (str): Short breed name (e.g., 'Chihuahua')
            - folder_name (str): Raw folder name (e.g., 'n02085620-Chihuahua')
            - class_index (int): Integer label (0 to 119), alphabetically assigned

        Raises:
            ValueError: If fewer than 2 breed classes are discovered.
        """
        logger.info("Scanning dataset at: %s", self.dataset_root)

        records: list[dict] = []
        skipped_folders: list[str] = []

        breed_dirs = sorted(
            d for d in self.dataset_root.iterdir()
            if d.is_dir() and d.name not in JUNK_FOLDERS
        )

        if len(breed_dirs) < 2:
            raise ValueError(
                f"Expected at least 2 breed directories, found {len(breed_dirs)}. "
                f"Check dataset path: {self.dataset_root}"
            )

        logger.info("Found %d breed directories.", len(breed_dirs))

        for breed_dir in breed_dirs:
            breed_name = self._extract_breed_name(breed_dir.name)
            image_files = [
                f for f in breed_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]

            if not image_files:
                skipped_folders.append(breed_dir.name)
                logger.warning("Empty breed directory (no images): %s", breed_dir.name)
                continue

            for img_path in image_files:
                records.append({
                    "filepath": str(img_path.resolve()),
                    "label": breed_name,
                    "folder_name": breed_dir.name,
                })

        if skipped_folders:
            logger.warning(
                "Skipped %d empty breed folder(s): %s",
                len(skipped_folders),
                skipped_folders,
            )

        df = pd.DataFrame(records)

        # Assign integer class indices sorted alphabetically by breed name
        # This is deterministic — the same breed always gets the same index
        unique_labels = sorted(df["label"].unique())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        df["class_index"] = df["label"].map(label_to_index)

        logger.info(
            "Dataset loaded: %d images across %d classes.",
            len(df),
            df["label"].nunique(),
        )
        return df

    def get_class_mapping(self) -> dict[str, int]:
        """
        Return the breed name → class index mapping.

        This mapping is saved to disk after dataset loading so that
        the inference engine can map model output indices back to breed names.

        Returns:
            Dictionary: { "Chihuahua": 0, "Afghan_hound": 1, ... }
        """
        df = self.load()
        return dict(zip(df["label"], df["class_index"], strict=True))

    @staticmethod
    def _extract_breed_name(folder_name: str) -> str:
        """
        Strip the ImageNet synset prefix from the folder name.

        Example:
            'n02085620-Chihuahua'  → 'Chihuahua'
            'n02088094-Afghan_hound' → 'Afghan_hound'
            'Chihuahua'            → 'Chihuahua'  (no prefix, returned as-is)

        Args:
            folder_name: Raw directory name from the dataset.

        Returns:
            Clean breed name string.
        """
        # Pattern: starts with 'n' followed by digits and a hyphen
        if "-" in folder_name and folder_name[0] == "n" and folder_name[1:9].isdigit():
            return folder_name.split("-", 1)[1]
        return folder_name
