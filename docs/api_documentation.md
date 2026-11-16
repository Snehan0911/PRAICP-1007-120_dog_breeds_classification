# CanineVision AI — REST API Documentation

The CanineVision AI inference server exposes asynchronous REST endpoints conforming to OpenAPI 3.1 specifications.

## Endpoints Summary

| Method | Path | Description | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/health` | Healthcheck & telemetry | None | `HealthResponse` |
| `GET` | `/model-info` | Architecture metadata | None | `ModelInfoResponse` |
| `POST` | `/predict` | Top-3 breed classification | Multipart Form (`file`) | `PredictionResponse` |
| `POST` | `/predict/explain` | Prediction + Grad-CAM | Multipart Form (`file`) | `ExplainResponse` |

---

### 1. GET `/health`
Returns runtime status and memory utilization.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_model": "mobilenet_v2",
  "num_classes": 120,
  "system_memory_percent": 68.4
}
```

---

### 2. POST `/predict`
Accepts a binary image file (`image/jpeg`, `image/png`, `image/webp`).

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog.jpg"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "predicted_breed": "Golden Retriever",
  "top_predictions": [
    {
      "breed": "Golden Retriever",
      "raw_breed": "n02099601-golden_retriever",
      "class_index": 56,
      "probability": 0.8842,
      "percentage": "88.42%"
    },
    {
      "breed": "Labrador Retriever",
      "raw_breed": "n02099712-Labrador_retriever",
      "class_index": 57,
      "probability": 0.0815,
      "percentage": "8.15%"
    },
    {
      "breed": "Chesapeake Bay Retriever",
      "raw_breed": "n02099849-Chesapeake_Bay_retriever",
      "class_index": 58,
      "probability": 0.0124,
      "percentage": "1.24%"
    }
  ],
  "confidence": {
    "level": "HIGH",
    "score": 0.8842,
    "is_reliable": true,
    "warning": null,
    "explanation": "Strong pattern alignment (88.4%) with significant confidence margin."
  },
  "inference_time_ms": 47.8,
  "model_version": "mobilenet_v2-v1.0"
}
```

---

### 3. POST `/predict/explain`
Includes the Grad-CAM heatmap visualization as a Base64-encoded JPEG in `gradcam_base64`.
