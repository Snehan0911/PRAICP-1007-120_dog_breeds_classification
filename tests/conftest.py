"""
CanineVision AI — Pytest Configuration and Shared Fixtures.

This file is loaded automatically by pytest before any test file.
Fixtures defined here are available to ALL test files without importing.

A fixture is a reusable setup function. Instead of calling DatasetLoader()
in every single test, you define it once here and pytest injects it automatically.

Usage in a test file:
    def test_something(dataset_df):
        assert len(dataset_df) > 0  # dataset_df is injected by pytest
"""

import sys
from pathlib import Path

import pytest

# Make src importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_paths(project_root):
    """Return a configured ProjectPaths instance."""
    from src.utils.paths import ProjectPaths
    return ProjectPaths(project_root=project_root)


@pytest.fixture(scope="session")
def sample_image_path(project_paths) -> Path:
    """
    Return the path to a real sample image from the dataset.
    Used to test image loading and preprocessing without mocking.
    """
    dataset_root = project_paths.dataset_root
    # Pick the first .jpg found in the first breed directory
    for breed_dir in sorted(dataset_root.iterdir()):
        if breed_dir.is_dir() and breed_dir.name != "Untitled Folder":
            for img_file in breed_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    return img_file
    pytest.skip("No sample image found in dataset — dataset may not be extracted")


@pytest.fixture(scope="session")
def small_dataset_df(project_paths):
    """
    Load a small subset of the dataset (5 images per class, 10 classes max).
    Fast to load; used for unit tests that don't need the full dataset.
    """
    from src.data.dataset_loader import DatasetLoader
    loader = DatasetLoader(dataset_root=project_paths.dataset_root)
    df = loader.load()
    # Sample 10 classes × 5 images each for speed
    classes = df["label"].unique()[:10]
    return (
        df[df["label"].isin(classes)]
        .groupby("label")
        .head(5)
        .reset_index(drop=True)
    )
