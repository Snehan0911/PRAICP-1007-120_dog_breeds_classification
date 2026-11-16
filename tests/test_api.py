"""
Tests — FastAPI Endpoints Integration Suite.
"""

import io

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient to trigger lifespan startup and shutdown."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 and model_loaded: True."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 120


def test_model_info_endpoint(client):
    """Verify /model-info returns 120 classes and proper input shape."""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_classes"] == 120
    assert data["input_shape"] == [224, 224, 3]


def test_predict_endpoint_valid_image(client, sample_image_path):
    """Verify POST /predict returns valid top-3 ranking and latency."""
    with open(sample_image_path, "rb") as f:
        files = {"file": ("sample.jpg", f, "image/jpeg")}
        resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "predicted_breed" in data
    assert len(data["top_predictions"]) == 3
    assert data["inference_time_ms"] > 0
    assert data["confidence"]["level"] in ["HIGH", "MEDIUM", "LOW", "UNCERTAIN"]


def test_predict_endpoint_invalid_file(client):
    """Verify POST /predict returns 400 when uploaded content is not an image."""
    files = {"file": ("fake.txt", io.BytesIO(b"this is not an image"), "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_explain_endpoint(client, sample_image_path):
    """Verify POST /predict/explain returns valid Grad-CAM base64 string."""
    with open(sample_image_path, "rb") as f:
        files = {"file": ("sample.jpg", f, "image/jpeg")}
        resp = client.post("/predict/explain", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["gradcam_base64"]) > 50  # Must contain base64 bytes
