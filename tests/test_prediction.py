"""
Unit tests for Image Validation and Prediction Engine.
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

from pathlib import Path

from src.inference.image_validator import ImageValidator
from src.inference.predictor import DogBreedPredictor


def test_image_validator_valid_sample():
    sample_path = Path("dog_photos/samples/golden_retriever_sample.jpg")
    assert sample_path.is_file(), "Sample golden retriever image must exist."

    result = ImageValidator.validate_file(sample_path)
    assert result.is_valid is True
    assert result.error_message is None
    assert result.width > 0
    assert result.height > 0
    assert result.format in {"JPEG", "JPG"}
    assert result.size_bytes > 0


def test_image_validator_nonexistent_file():
    result = ImageValidator.validate_file("dog_photos/samples/nonexistent_dog.jpg")
    assert result.is_valid is False
    assert "not found" in result.error_message.lower()


def test_image_validator_unsupported_extension(tmp_path):
    fake_txt = tmp_path / "notes.txt"
    fake_txt.write_text("Not an image file.")

    result = ImageValidator.validate_file(fake_txt)
    assert result.is_valid is False
    assert "unsupported file extension" in result.error_message.lower()


def test_image_validator_corrupt_bytes():
    corrupt_bytes = b"GIF89a corrupted header garbage bytes"
    result = ImageValidator.validate_bytes(corrupt_bytes, filename="test.jpg")
    assert result.is_valid is False
    assert result.error_message is not None


def test_image_validator_bytes_valid():
    sample_path = Path("dog_photos/samples/Chihuahua_sample.jpg")
    with open(sample_path, "rb") as f:
        data = f.read()

    result = ImageValidator.validate_bytes(data, filename="Chihuahua_sample.jpg")
    assert result.is_valid is True
    assert result.width > 0
    assert result.height > 0


def test_predictor_inference_and_ranking():
    model_path = Path("model/saved_models/best_model.keras")
    mapping_path = Path("model/class_mapping/class_indices.json")
    sample_path = Path("dog_photos/samples/German_shepherd_sample.jpg")

    predictor = DogBreedPredictor(
        model_path=model_path,
        class_mapping_path=mapping_path,
        enable_gradcam=True,
    )

    # Predict top 3
    result = predictor.predict(sample_path, top_k=3, explain=True)

    assert result.top_prediction is not None
    assert result.top_prediction.rank == 1
    assert len(result.top_k_predictions) == 3
    assert result.latency_ms > 0.0
    assert result.image_shape == (224, 224, 3)

    # Probabilities in descending order
    probs = [p.probability for p in result.top_k_predictions]
    assert probs == sorted(probs, reverse=True)

    # Grad-CAM generated
    assert result.gradcam_base64 is not None
    assert result.gradcam_base64.startswith("data:image/png;base64,")

    # Dict serialization
    d = result.to_dict()
    assert "top_prediction" in d
    assert "latency_ms" in d
    assert d["latency_ms"] > 0


def test_predictor_predict_file_with_validation():
    predictor = DogBreedPredictor(enable_gradcam=False)
    sample_path = Path("dog_photos/samples/Samoyed_sample.jpg")

    is_valid, output, error = predictor.predict_file_with_validation(sample_path, top_k=5)
    assert is_valid is True
    assert error is None
    assert output is not None
    assert len(output.top_k_predictions) == 5

    # Nonexistent file test
    is_valid_fake, output_fake, error_fake = predictor.predict_file_with_validation(
        "dog_photos/samples/does_not_exist.jpg"
    )
    assert is_valid_fake is False
    assert output_fake is None
    assert error_fake is not None
