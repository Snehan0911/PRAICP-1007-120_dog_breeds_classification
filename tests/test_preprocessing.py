"""
Unit tests for image loading, validation, and preprocessing normalizations.
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.data.preprocessing import (
    get_preprocessor,
    load_and_validate_image,
    preprocess_custom_cnn,
    preprocess_mobilenet_v2,
)


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Creates a temporary valid RGB JPEG image."""
    img_path = tmp_path / "test_dog.jpg"
    img = Image.new("RGB", (300, 400), color=(128, 64, 32))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def corrupt_image_path(tmp_path: Path) -> Path:
    """Creates a temporary corrupted image file."""
    bad_path = tmp_path / "corrupt_dog.jpg"
    bad_path.write_bytes(b"NOT_A_VALID_JPEG_HEADER_1234567890")
    return bad_path


def test_load_and_validate_image_valid(sample_image_path: Path):
    arr = load_and_validate_image(sample_image_path, target_size=(224, 224))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8


def test_load_and_validate_image_missing():
    with pytest.raises(ValueError, match="Failed to load or validate"):
        load_and_validate_image("non_existent_dog_image_99999.jpg")


def test_load_and_validate_image_corrupt(corrupt_image_path: Path):
    with pytest.raises(ValueError, match="Failed to load or validate"):
        load_and_validate_image(corrupt_image_path)


def test_preprocess_custom_cnn():
    dummy = np.full((224, 224, 3), 255, dtype=np.uint8)
    tensor = preprocess_custom_cnn(dummy)
    assert tensor.dtype == tf.float32
    assert float(tf.reduce_max(tensor)) == pytest.approx(1.0, abs=1e-5)
    assert float(tf.reduce_min(tensor)) == pytest.approx(1.0, abs=1e-5)


def test_preprocess_mobilenet_v2():
    dummy = np.full((224, 224, 3), 255, dtype=np.uint8)
    tensor = preprocess_mobilenet_v2(dummy)
    assert tensor.dtype == tf.float32
    assert float(tf.reduce_max(tensor)) == pytest.approx(1.0, abs=1e-5)

    dummy_zero = np.zeros((224, 224, 3), dtype=np.uint8)
    tensor_zero = preprocess_mobilenet_v2(dummy_zero)
    assert float(tf.reduce_min(tensor_zero)) == pytest.approx(-1.0, abs=1e-5)


def test_get_preprocessor_dispatch():
    custom_fn = get_preprocessor("custom_cnn")
    mobilenet_fn = get_preprocessor("mobilenet_v2")
    assert callable(custom_fn)
    assert callable(mobilenet_fn)

    with pytest.raises(ValueError, match="Unknown model_name"):
        get_preprocessor("unsupported_resnet_model")
