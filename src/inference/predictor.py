"""
Inference and Prediction Engine for Dog Breeds Classification.
Project: Dog Breeds Classification (PRAICP-1007-120)

Provides high-performance inference, Top-K probability ranking,
inference latency benchmarking, and optional Grad-CAM explainability.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    load_and_validate_image,
    preprocess_mobilenet_v2,
)
from src.inference.gradcam import GradCAMExplainer
from src.inference.image_validator import ImageValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Predictor")


@dataclass
class BreedPrediction:
    """Individual breed prediction candidate."""
    rank: int
    breed: str
    display_name: str
    probability: float
    percentage: float


@dataclass
class PredictionOutput:
    """Consolidated inference output payload."""
    top_prediction: BreedPrediction
    top_k_predictions: list[BreedPrediction]
    latency_ms: float
    image_shape: tuple[int, int, int]
    gradcam_base64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts the prediction output to a JSON-serializable dictionary."""
        return {
            "top_prediction": asdict(self.top_prediction),
            "top_k_predictions": [asdict(p) for p in self.top_k_predictions],
            "latency_ms": round(self.latency_ms, 2),
            "image_shape": list(self.image_shape),
            "gradcam_base64": self.gradcam_base64,
        }


def format_breed_display_name(raw_name: str) -> str:
    """
    Formats a raw class name into a clean, human-readable display string.

    Examples:
        'Chihuahua' -> 'Chihuahua'
        'german_shepherd' -> 'German Shepherd'
        'rhodesian_ridgeback' -> 'Rhodesian Ridgeback'
    """
    cleaned = raw_name.replace("_", " ").strip()
    return " ".join(word.capitalize() for word in cleaned.split())


class DogBreedPredictor:
    """
    Production-grade inference engine for the Dog Breeds Classification model.
    """

    DEFAULT_MODEL_PATH = Path("model/saved_models/best_model.keras")
    DEFAULT_CLASS_MAPPING_PATH = Path("model/class_mapping/class_indices.json")

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_mapping_path: str | Path | None = None,
        model: tf.keras.Model | None = None,
        enable_gradcam: bool = True,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        Initializes the DogBreedPredictor engine.

        Args:
            model_path: Path to the serialized Keras model file.
            class_mapping_path: Path to the class index JSON mapping.
            model: Pre-loaded Keras model (useful for unit testing or custom injection).
            enable_gradcam: Whether to initialize Grad-CAM explainability.
            target_size: Target image dimensions (width, height) expected by model.
            preprocess_fn: Custom preprocessing function; defaults to MobileNetV2 normalization.
        """
        self.target_size = target_size
        self.preprocess_fn = preprocess_fn or preprocess_mobilenet_v2

        # 1. Load or bind Model
        if model is not None:
            self.model = model
            logger.info("Initialized DogBreedPredictor with pre-loaded model instance.")
        else:
            resolved_model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
            if not resolved_model_path.is_file():
                raise FileNotFoundError(
                    f"Trained model not found at {resolved_model_path}. "
                    "Please run model training (Milestone 8) before running inference."
                )
            logger.info("Loading model from %s...", resolved_model_path)
            self.model = tf.keras.models.load_model(str(resolved_model_path), compile=False)
            logger.info("Model loaded successfully (%d layers).", len(self.model.layers))

        # 2. Load Class Mapping
        resolved_mapping_path = (
            Path(class_mapping_path) if class_mapping_path else self.DEFAULT_CLASS_MAPPING_PATH
        )
        if not resolved_mapping_path.is_file():
            raise FileNotFoundError(
                f"Class indices mapping not found at {resolved_mapping_path}. "
                "Ensure class_indices.json exists."
            )

        with open(resolved_mapping_path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)

        # Invert to {int_index: class_name}
        if all(k.isdigit() for k in raw_mapping):
            self.index_to_class = {int(k): str(v) for k, v in raw_mapping.items()}
        else:
            self.index_to_class = {int(v): str(k) for k, v in raw_mapping.items()}

        self.num_classes = len(self.index_to_class)
        logger.info("Loaded class mapping with %d classes.", self.num_classes)

        # 3. Optional Grad-CAM Explainer
        self.explainer: GradCAMExplainer | None = None
        if enable_gradcam:
            try:
                self.explainer = GradCAMExplainer(self.model)
                logger.info(
                    "Grad-CAM explainer attached to target layer '%s'.",
                    self.explainer.layer_name,
                )
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                logger.warning("Could not initialize GradCAMExplainer: %s", e)
                self.explainer = None

    def predict(
        self,
        image_input: str | Path | bytes | np.ndarray,
        top_k: int = 3,
        explain: bool = True,
    ) -> PredictionOutput:
        """
        Executes prediction on an input image, returning Top-K predicted dog breeds.

        Args:
            image_input: Filepath, bytes, or RGB uint8 numpy array.
            top_k: Number of highest-confidence predictions to return (default: 3).
            explain: Whether to generate Grad-CAM visualization.

        Returns:
            PredictionOutput with top predictions, latency, and Grad-CAM base64 image.

        Raises:
            ValueError: If the image cannot be loaded or is invalid.
        """
        # Step 1: Image loading and standard RGB resizing
        if isinstance(image_input, np.ndarray):
            if image_input.ndim != 3 or image_input.shape[2] != 3:
                raise ValueError(f"Expected 3-channel RGB image array, got shape {image_input.shape}")
            if image_input.shape[0] != self.target_size[1] or image_input.shape[1] != self.target_size[0]:
                image_rgb = tf.image.resize(image_input, self.target_size).numpy().astype(np.uint8)
            else:
                image_rgb = image_input.astype(np.uint8)
        else:
            image_rgb = load_and_validate_image(image_input, target_size=self.target_size)

        # Step 2: Architecture-specific normalization and batch dimension
        img_preprocessed = self.preprocess_fn(image_rgb)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Step 3: Run inference and measure latency
        start_time = time.perf_counter()
        raw_predictions = self.model.predict(img_batch, verbose=0)[0]
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        # Step 4: Extract Top-K predictions
        # Handle softmax vs logits
        if not np.isclose(np.sum(raw_predictions), 1.0, atol=1e-2):
            probabilities = tf.nn.softmax(raw_predictions).numpy()
        else:
            probabilities = raw_predictions

        # Sort indices by descending probability
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        candidates: list[BreedPrediction] = []
        for rank, idx in enumerate(top_indices, start=1):
            prob = float(probabilities[idx])
            raw_breed = self.index_to_class.get(idx, f"Breed #{idx}")
            display_name = format_breed_display_name(raw_breed)
            candidates.append(
                BreedPrediction(
                    rank=rank,
                    breed=raw_breed,
                    display_name=display_name,
                    probability=round(prob, 4),
                    percentage=round(prob * 100.0, 2),
                )
            )

        top_prediction = candidates[0]

        # Step 5: Optional Grad-CAM visual explanation
        gradcam_b64: str | None = None
        if explain and self.explainer is not None:
            try:
                top_class_idx = int(top_indices[0])
                explanation = self.explainer.explain_image(
                    image_rgb,
                    class_idx=top_class_idx,
                    preprocess_fn=self.preprocess_fn,
                )
                gradcam_b64 = f"data:image/png;base64,{explanation['overlay_base64']}"
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                logger.warning("Grad-CAM generation failed during inference: %s", e)
                gradcam_b64 = None

        return PredictionOutput(
            top_prediction=top_prediction,
            top_k_predictions=candidates,
            latency_ms=latency_ms,
            image_shape=image_rgb.shape,
            gradcam_base64=gradcam_b64,
        )

    def predict_file_with_validation(
        self,
        file_path: str | Path,
        top_k: int = 3,
        explain: bool = True,
    ) -> tuple[bool, PredictionOutput | None, str | None]:
        """
        Validates an image file and executes prediction if valid.

        Returns:
            Tuple of (is_valid, prediction_output, error_message).
        """
        validation = ImageValidator.validate_file(file_path)
        if not validation.is_valid:
            return False, None, validation.error_message

        try:
            prediction = self.predict(file_path, top_k=top_k, explain=explain)
            return True, prediction, None
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Inference failed on file %s: %s", file_path, e)
            return False, None, str(e)
