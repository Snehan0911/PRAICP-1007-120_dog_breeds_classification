"""
Tests — Dataset discovery and class mapping.

These tests verify that:
1. The dataset loader finds the correct number of breeds (120)
2. The correct total image count is discovered
3. Breed names are correctly extracted from folder names
4. Class indices are assigned deterministically
5. The class mapping is consistent across runs
"""



class TestDatasetLoader:
    """Unit tests for the DatasetLoader class."""

    def test_dataset_root_exists(self, project_paths):
        """The dataset root directory must exist before anything else."""
        assert project_paths.dataset_root.exists(), (
            f"Dataset root not found: {project_paths.dataset_root}\n"
            "Please extract the dataset before running tests."
        )

    def test_correct_number_of_classes(self, project_paths):
        """Expect exactly 120 breed directories."""
        from src.data.dataset_loader import DatasetLoader

        loader = DatasetLoader(dataset_root=project_paths.dataset_root)
        df = loader.load()

        num_classes = df["label"].nunique()
        assert num_classes == 120, (
            f"Expected 120 breed classes, found {num_classes}. "
            "Some breed directories may be missing or misnamed."
        )

    def test_total_image_count_in_range(self, project_paths):
        """Total image count should be in a realistic range (15,000–25,000)."""
        from src.data.dataset_loader import DatasetLoader

        loader = DatasetLoader(dataset_root=project_paths.dataset_root)
        df = loader.load()

        total = len(df)
        assert 15_000 <= total <= 25_000, (
            f"Expected 15,000–25,000 images, found {total}. "
            "Dataset may be corrupted or incomplete."
        )

    def test_breed_name_extraction(self):
        """Synset prefix should be stripped from folder names correctly."""
        from src.data.dataset_loader import DatasetLoader

        assert DatasetLoader._extract_breed_name("n02085620-Chihuahua") == "Chihuahua"
        assert DatasetLoader._extract_breed_name("n02088094-Afghan_hound") == "Afghan_hound"
        assert DatasetLoader._extract_breed_name("Chihuahua") == "Chihuahua"  # No prefix

    def test_class_indices_are_zero_based_and_contiguous(self, project_paths):
        """Class indices should be 0, 1, 2, ..., 119 with no gaps."""
        from src.data.dataset_loader import DatasetLoader

        loader = DatasetLoader(dataset_root=project_paths.dataset_root)
        df = loader.load()

        indices = sorted(df["class_index"].unique())
        expected = list(range(len(indices)))
        assert indices == expected, "Class indices are not contiguous from 0."

    def test_class_mapping_is_deterministic(self, project_paths):
        """Running load() twice must produce the same class mapping."""
        from src.data.dataset_loader import DatasetLoader

        loader = DatasetLoader(dataset_root=project_paths.dataset_root)
        df1 = loader.load()
        df2 = loader.load()

        mapping1 = dict(zip(df1["label"], df1["class_index"], strict=False))
        mapping2 = dict(zip(df2["label"], df2["class_index"], strict=False))

        assert mapping1 == mapping2, (
            "Class mapping is not deterministic — breed indices differ between runs."
        )

    def test_no_empty_breed_directories_in_dataframe(self, project_paths):
        """Every breed in the DataFrame should have at least 1 image."""
        from src.data.dataset_loader import DatasetLoader

        loader = DatasetLoader(dataset_root=project_paths.dataset_root)
        df = loader.load()

        counts = df.groupby("label").size()
        empty = counts[counts == 0]
        assert len(empty) == 0, f"Breeds with 0 images: {empty.index.tolist()}"

    def test_all_filepaths_exist(self, small_dataset_df):
        """Every filepath in the sample DataFrame should point to a real file."""
        from pathlib import Path

        missing = [
            row["filepath"] for _, row in small_dataset_df.iterrows()
            if not Path(row["filepath"]).exists()
        ]
        assert len(missing) == 0, f"Missing files: {missing[:5]}"
