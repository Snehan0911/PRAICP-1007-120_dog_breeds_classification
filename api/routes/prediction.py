"""
CanineVision AI — Prediction & Explainability Routes.
"""

import base64

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.schemas.response import ExplainResponse, PredictionResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Inference & Explainability"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_breed(
    request: Request,
    file: UploadFile = File(..., description="Uploaded dog photograph (JPEG, PNG, WEBP)"),
) -> PredictionResponse:
    """
    Classifies a dog image into one of 120 breeds.
    Returns predicted breed, top-3 candidates, probability scores, and latency.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction engine is not ready.")

    image_bytes = await file.read()
    result = predictor.predict(image_bytes, top_k=3, return_gradcam=False)

    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("message", "Inference failed."))

    return PredictionResponse(
        status="success",
        predicted_breed=result["predicted_breed"],
        top_predictions=result["top_predictions"],
        confidence=result["confidence"],
        inference_time_ms=result["inference_time_ms"],
        model_version=result["model_version"],
    )


@router.post("/predict/explain", response_model=ExplainResponse)
async def predict_and_explain(
    request: Request,
    file: UploadFile = File(..., description="Uploaded dog photograph (JPEG, PNG, WEBP)"),
) -> ExplainResponse:
    """
    Classifies a dog image and generates a Grad-CAM explainability heatmap overlay (Base64 JPEG).
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction engine is not ready.")

    image_bytes = await file.read()
    result = predictor.predict(image_bytes, top_k=3, return_gradcam=True)

    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("message", "Inference failed."))

    # Encode Grad-CAM RGB overlay to base64 JPEG
    overlay = result.get("gradcam_overlay")
    if overlay is not None:
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", bgr)
        b64_str = base64.b64encode(buffer).decode("utf-8")
    else:
        b64_str = ""

    return ExplainResponse(
        status="success",
        predicted_breed=result["predicted_breed"],
        top_predictions=result["top_predictions"],
        confidence=result["confidence"],
        inference_time_ms=result["inference_time_ms"],
        model_version=result["model_version"],
        gradcam_base64=b64_str,
    )
