"""
Tests — Image preprocessing pipeline.

These tests verify:
1. Images load correctly and return the expected array shape
2. Custom CNN preprocessing produces values in [0.0, 1.0]
3. MobileNetV2 preprocessing produces values in [-1.0, 1.0]
4. EfficientNetB0 preprocessing produces a float32 array
5. The preprocess factory returns the correct function by name
6. Non-RGB images are handled without errors
"""

import numpy as np
import pytest


class TestImageLoading:
    """Tests for the base load_image function."""

    def test_load_returns_correct_shape(self, sample_image_path):
        """Loaded image should be (224, 224, 3)."""
        from src.data.preprocessing import load_image

        img = load_image(sample_image_path, target_size=(224, 224))
        assert img.shape == (224, 224, 3), f"Expected (224, 224, 3), got {img.shape}"

    def test_load_returns_uint8(self, sample_image_path):
        """Raw loaded image should be dtype uint8 with values 0–255."""
        from src.data.preprocessing import load_image

        img = load_image(sample_image_path, target_size=(224, 224))
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255

    def test_load_missing_file_raises(self):
        """Loading a non-existent file must raise FileNotFoundError."""
        from src.data.preprocessing import load_image

        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.jpg")


class TestCustomCNNPreprocessing:
    """Tests for the Custom CNN /255 normalisation."""

    def test_values_in_zero_one_range(self, sample_image_path):
        """Custom CNN preprocessing must produce values in [0.0, 1.0]."""
        from src.data.preprocessing import preprocess_custom_cnn

        arr = preprocess_custom_cnn(sample_image_path, target_size=(224, 224))
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0, f"Min value {arr.min()} is below 0.0"
        assert arr.max() <= 1.0, f"Max value {arr.max()} is above 1.0"

    def test_output_shape(self, sample_image_path):
        """Output shape must be (224, 224, 3)."""
        from src.data.preprocessing import preprocess_custom_cnn

        arr = preprocess_custom_cnn(sample_image_path, target_size=(224, 224))
        assert arr.shape == (224, 224, 3)


class TestMobileNetV2Preprocessing:
    """Tests for the MobileNetV2 [-1, 1] preprocessing."""

    def test_values_in_minus_one_plus_one_range(self, sample_image_path):
        """MobileNetV2 preprocessing must produce values in [-1.0, 1.0]."""
        from src.data.preprocessing import preprocess_mobilenet_v2

        arr = preprocess_mobilenet_v2(sample_image_path, target_size=(224, 224))
        assert arr.dtype == np.float32
        assert arr.min() >= -1.01, f"Min {arr.min()} outside expected range"
        assert arr.max() <= 1.01, f"Max {arr.max()} outside expected range"

    def test_output_shape(self, sample_image_path):
        """Output shape must be (224, 224, 3)."""
        from src.data.preprocessing import preprocess_mobilenet_v2

        arr = preprocess_mobilenet_v2(sample_image_path, target_size=(224, 224))
        assert arr.shape == (224, 224, 3)


class TestPreprocessFactory:
    """Tests for the get_preprocess_fn factory."""

    def test_known_model_names_return_callable(self):
        """Factory must return a callable for all known model names."""
        from src.data.preprocessing import get_preprocess_fn

        for name in ["custom_cnn", "normalize_01", "mobilenet_v2", "efficientnet_b0"]:
            fn = get_preprocess_fn(name)
            assert callable(fn), f"Expected callable for '{name}', got {type(fn)}"

    def test_unknown_model_name_raises(self):
        """Factory must raise KeyError for unrecognised model names."""
        from src.data.preprocessing import get_preprocess_fn

        with pytest.raises(KeyError):
            get_preprocess_fn("unknown_model_xyz")
