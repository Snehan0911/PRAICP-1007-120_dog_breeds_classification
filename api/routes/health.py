"""
CanineVision AI — Health Check Endpoint.
"""

import psutil
from fastapi import APIRouter, Request

from api.schemas.response import HealthResponse

router = APIRouter(tags=["System Health"])


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """
    Returns API health status, loaded model readiness, and memory telemetry.
    """
    predictor = getattr(request.app.state, "predictor", None)
    is_loaded = predictor is not None and predictor.model is not None
    model_name = predictor.model_name if predictor else "none"
    num_classes = len(predictor.index_to_breed) if predictor else 0

    mem = psutil.virtual_memory()

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        active_model=model_name,
        num_classes=num_classes,
        system_memory_percent=round(mem.percent, 1),
    )
