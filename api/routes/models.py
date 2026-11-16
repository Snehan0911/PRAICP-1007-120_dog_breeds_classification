"""
CanineVision AI — Model Information Endpoint.
"""

from fastapi import APIRouter, Request

from api.schemas.response import ModelInfoResponse

router = APIRouter(tags=["Model Metadata"])


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(request: Request) -> ModelInfoResponse:
    """
    Returns architecture specifications, input dimensions, and supported breed classes.
    """
    predictor = getattr(request.app.state, "predictor", None)
    model_name = predictor.model_name if predictor else "mobilenet_v2"
    class_names = [predictor.index_to_breed[i] for i in sorted(predictor.index_to_breed.keys())] if predictor else []

    return ModelInfoResponse(
        model_name=model_name,
        version="1.0.0",
        input_shape=[224, 224, 3],
        num_classes=len(class_names),
        preprocessing=f"{model_name}_preprocessing",
        class_names=class_names,
    )
