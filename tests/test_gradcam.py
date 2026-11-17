"""
Unit tests for Grad-CAM explainability pipeline.
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

from pathlib import Path

import numpy as np

from src.inference.gradcam import GradCAMExplainer
from src.models.custom_cnn import build_custom_cnn


def test_gradcam_explainer_layer_discovery():
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    explainer = GradCAMExplainer(model)
    assert explainer.layer_name == "conv4_1"
    assert explainer.gradient_model is not None


def test_gradcam_heatmap_dimensions_and_bounds():
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    explainer = GradCAMExplainer(model, layer_name="conv4_1")

    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    heatmap = explainer.compute_heatmap(dummy_input, class_idx=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0


def test_gradcam_overlay_generation():
    dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
    dummy_heatmap = np.full((224, 224), 0.5, dtype=np.float32)

    overlay = GradCAMExplainer.overlay_heatmap(dummy_img, dummy_heatmap, alpha=0.4)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_gradcam_explain_image_with_sample():
    sample_path = Path("dog_photos/samples/Chihuahua_sample.jpg")
    if sample_path.exists():
        model_path = Path("model/saved_models/best_model.keras")
        if model_path.exists():
            explainer = GradCAMExplainer(model_path)
            res = explainer.explain_image(sample_path)
            assert "predicted_class_id" in res
            assert "confidence" in res
            assert "overlay_base64" in res
            assert len(res["overlay_base64"]) > 100
