"""
CanineVision AI — FastAPI Pydantic Response Schemas.
"""


from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    breed: str = Field(..., description="Clean breed name (e.g. 'Golden Retriever')")
    raw_breed: str = Field(..., description="Raw dataset folder synset name")
    class_index: int = Field(..., description="Deterministic zero-based class index")
    probability: float = Field(..., description="Softmax probability score (0.0 to 1.0)")
    percentage: str = Field(..., description="Formatted percentage string")


class ConfidenceInfo(BaseModel):
    level: str = Field(..., description="Tier: HIGH | MEDIUM | LOW | UNCERTAIN")
    score: float = Field(..., description="Top-1 probability score")
    is_reliable: bool = Field(..., description="Whether inference meets reliability thresholds")
    warning: str | None = Field(None, description="Actionable warning message if applicable")
    explanation: str = Field(..., description="Contextual technical reasoning")


class PredictionResponse(BaseModel):
    status: str = Field("success", description="Response status ('success' or 'error')")
    predicted_breed: str = Field(..., description="Top-1 predicted dog breed")
    top_predictions: list[TopPrediction] = Field(..., description="Top-3 ranked predictions")
    confidence: ConfidenceInfo = Field(..., description="Calibrated confidence evaluation")
    inference_time_ms: float = Field(..., description="Server-side inference execution latency in milliseconds")
    model_version: str = Field(..., description="Model identifier and version tag")


class ExplainResponse(PredictionResponse):
    gradcam_base64: str = Field(..., description="Base64 encoded JPEG of Grad-CAM explainability heatmap overlay")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health state")
    model_loaded: bool = Field(..., description="Whether deep learning model is resident in RAM")
    active_model: str = Field(..., description="Loaded model architecture name")
    num_classes: int = Field(..., description="Supported dog breed categories (120)")
    system_memory_percent: float = Field(..., description="Current system RAM utilization percentage")


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    input_shape: list[int]
    num_classes: int
    preprocessing: str
    class_names: list[str]
