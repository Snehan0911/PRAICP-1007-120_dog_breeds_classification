"""
Web Application and REST API Integration Tests.
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

import io
from pathlib import Path

import pytest

from app import app


@pytest.fixture
def client():
    """Configures a Flask test client with testing mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Dog Breeds Classification" in html
    assert "PRAICP-1007-120" in html
    assert "Upload Dog Photograph" in html


def test_dashboard_route(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "20,580" in html
    assert "120" in html
    assert "MobileNetV2" in html


def test_analytics_route(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Confusion Matrix" in html
    assert "Top-1 Accuracy" in html
    assert "33.00%" in html


def test_model_info_route(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "MobileNetV2" in html
    assert "out_relu" in html
    assert "Inverted Residual" in html


def test_about_route(client):
    response = client.get("/about")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "PRAICP-1007-120" in html
    assert "Stanford Dogs" in html


def test_health_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["project"] == "Dog Breeds Classification"
    assert data["classes"] == 120


def test_predict_with_sample_image(client):
    response = client.post("/predict", data={"sample_name": "golden_retriever_sample.jpg"})
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Top Predicted Breed" in html
    assert "Confidence Score" in html
    assert "Top 3 Predictions" in html
    assert "data:image/png;base64," in html


def test_predict_without_file_redirects(client):
    response = client.post("/predict", data={})
    assert response.status_code == 302  # Redirects back to index


def test_api_predict_success(client):
    sample_path = Path("dog_photos/samples/Chihuahua_sample.jpg")
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "image": (io.BytesIO(img_bytes), "Chihuahua_sample.jpg"),
        "top_k": "3",
        "explain": "true",
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert "data" in payload
    assert "top_prediction" in payload["data"]
    assert len(payload["data"]["top_k_predictions"]) == 3
    assert payload["data"]["latency_ms"] > 0


def test_api_predict_missing_file(client):
    response = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["status"] == "error"
