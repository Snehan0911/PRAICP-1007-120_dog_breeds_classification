"""
Inference, prediction engine, image validation, and Grad-CAM explainability.
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

from .gradcam import GradCAMExplainer
from .image_validator import ImageValidator, ValidationResult
from .predictor import BreedPrediction, DogBreedPredictor, PredictionOutput

__all__ = [
    "BreedPrediction",
    "DogBreedPredictor",
    "GradCAMExplainer",
    "ImageValidator",
    "PredictionOutput",
    "ValidationResult",
]
