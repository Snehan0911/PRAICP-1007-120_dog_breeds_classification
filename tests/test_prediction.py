"""
Tests — Prediction Engine, Image Validation, and Confidence Interpretation.
"""


from src.inference.confidence import ConfidenceInterpreter
from src.inference.image_validator import ImageValidator
from src.inference.predictor import BreedPredictor


def test_confidence_interpreter_tiers():
    """Verify confidence levels and warnings."""
    interpreter = ConfidenceInterpreter(high_threshold=0.70, medium_threshold=0.40, uncertain_threshold=0.20)

    # High
    rep_high = interpreter.interpret([0.85, 0.10, 0.05])
    assert rep_high.level == "HIGH"
    assert rep_high.is_reliable is True

    # Medium
    rep_med = interpreter.interpret([0.55, 0.25, 0.20])
    assert rep_med.level == "MEDIUM"

    # Low
    rep_low = interpreter.interpret([0.30, 0.28, 0.20])
    assert rep_low.level == "LOW"

    # Uncertain / OOD warning
    rep_unc = interpreter.interpret([0.15, 0.14, 0.12])
    assert rep_unc.level == "UNCERTAIN"
    assert rep_unc.warning is not None


def test_image_validator_corrupt_detection():
    """Verify corrupt or empty bytes are rejected."""
    valid, msg, _ = ImageValidator.validate_bytes(b"")
    assert not valid
    assert "empty" in msg.lower()

    valid, msg, _ = ImageValidator.validate_bytes(b"not_an_image_bytes_here")
    assert not valid


def test_predictor_end_to_end_on_sample(sample_image_path):
    """Verify end-to-end prediction returns valid schema and top 3 results."""
    predictor = BreedPredictor(model_name="mobilenet_v2")
    res = predictor.predict(sample_image_path, top_k=3, return_gradcam=True)

    assert res["status"] == "success"
    assert "predicted_breed" in res
    assert len(res["top_predictions"]) == 3
    assert res["inference_time_ms"] > 0
    assert res["confidence"]["level"] in ["HIGH", "MEDIUM", "LOW", "UNCERTAIN"]
    assert res["gradcam_overlay"] is not None
