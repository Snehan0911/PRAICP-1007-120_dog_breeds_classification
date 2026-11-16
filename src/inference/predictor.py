"""
CanineVision AI — Production Predictor Engine.

Core Inference Service:
- Persistent model caching in memory (loads once at startup).
- Validates input images against corruption, size, and dimensions.
- Applies model-specific preprocessing.
- High-precision latency profiling (wall-clock inference time in ms).
- Top-3 breed ranking with confidence interpretation.
- Optional Grad-CAM explainability generation.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.preprocessing import get_preprocess_fn
from src.inference.confidence import ConfidenceInterpreter
from src.inference.gradcam import GradCAMExplainer
from src.inference.image_validator import ImageValidator
from src.models.model_factory import get_model
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class BreedPredictor:
    """
    Thread-safe, high-performance dog breed inference engine.

    Args:
        model_path: Path to saved .keras model weights. If None or not found,
                    falls back to instantiating MobileNetV2 with ImageNet features.
        class_mapping_path: Path to class_indices.json.
        model_name: Architecture name ('mobilenet_v2', 'custom_cnn', etc.).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_mapping_path: str | Path | None = None,
        model_name: str = "mobilenet_v2",
    ) -> None:
        self.paths = ProjectPaths()
        self.model_name = model_name.lower()
        self.confidence_interpreter = ConfidenceInterpreter()
        self.preprocess_fn = get_preprocess_fn(self.model_name)

        # 1. Load class mapping
        mapping_file = Path(class_mapping_path) if class_mapping_path else self.paths.artifacts_class_mapping / "class_indices.json"
        if mapping_file.exists():
            with open(mapping_file, encoding="utf-8") as f:
                self.class_indices = json.load(f)
            # Invert mapping: integer index -> breed name
            self.index_to_breed = {v: k for k, v in self.class_indices.items()}
        else:
            logger.warning("Class mapping not found at %s. Initializing dummy 120-class mapping.", mapping_file)
            self.index_to_breed = {i: f"Breed_{i}" for i in range(120)}

        # 2. Load model
        model_file = Path(model_path) if model_path else self.paths.artifacts_models / f"{self.model_name}_best.keras"
        if model_file.exists():
            logger.info("Loading pre-trained model checkpoint from: %s", model_file)
            self.model = tf.keras.models.load_model(model_file)
        else:
            logger.info("Checkpoint %s not found. Instantiating fresh %s model...", model_file, self.model_name)
            self.model = get_model(self.model_name, num_classes=len(self.index_to_breed))

        # 3. Grad-CAM explainer
        try:
            self.explainer = GradCAMExplainer(self.model)
        except Exception as e:
            logger.warning("Grad-CAM explainer could not be initialized: %s", e)
            self.explainer = None

        logger.info("BreedPredictor ready. Model: %s, Classes: %d", self.model_name, len(self.index_to_breed))

    def predict(
        self,
        image_input: str | Path | bytes | Image.Image,
        top_k: int = 3,
        return_gradcam: bool = False,
    ) -> dict[str, Any]:
        """
        Runs complete inference pipeline on an image.

        Args:
            image_input: File path, raw bytes, or PIL Image.
            top_k: Number of highest ranking predictions to return.
            return_gradcam: Whether to generate Grad-CAM heatmap array.

        Returns:
            Dictionary matching the prediction response schema.
        """
        # Step 1: Input conversion to PIL Image & Validation
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                img_bytes = f.read()
            valid, err, pil_img = ImageValidator.validate_bytes(img_bytes)
            str(image_input)
        elif isinstance(image_input, bytes):
            valid, err, pil_img = ImageValidator.validate_bytes(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            valid, err = True, None
        else:
            return {"status": "error", "message": f"Unsupported input type: {type(image_input)}"}

        if not valid or pil_img is None:
            return {"status": "error", "message": err or "Invalid image."}

        # Step 2: Preprocess for Model
        resized = pil_img.resize((224, 224), Image.LANCZOS)
        arr = np.array(resized, dtype=np.float32)

        if "mobilenet" in self.model_name:
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        elif "efficientnet" in self.model_name:
            arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        else:
            arr = arr / 255.0

        batch_input = np.expand_dims(arr, axis=0)  # Shape (1, 224, 224, 3)

        # Step 3: Run Inference & Profile Latency
        start_time = time.perf_counter()
        probabilities = self.model.predict(batch_input, verbose=0)[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 4: Top-K Ranking
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            breed_label = self.index_to_breed.get(int(idx), f"Unknown_{idx}")
            clean_name = breed_label.replace("_", " ").title()
            top_predictions.append({
                "breed": clean_name,
                "raw_breed": breed_label,
                "class_index": int(idx),
                "probability": round(float(probabilities[idx]), 4),
                "percentage": f"{probabilities[idx] * 100:.2f}%",
            })

        # Step 5: Confidence Interpretation
        top_probs = [p["probability"] for p in top_predictions]
        conf_report = self.confidence_interpreter.interpret(top_probs)

        # Step 6: Optional Grad-CAM Explainability
        gradcam_overlay = None
        if return_gradcam and self.explainer is not None:
            try:
                top_class = int(top_indices[0])
                heatmap = self.explainer.compute_heatmap(batch_input, class_index=top_class)
                # Generate overlay from PIL Image
                h, w = pil_img.size[1], pil_img.size[0]
                import cv2
                resized_heat = cv2.resize(heatmap, (w, h))
                uint8_heat = np.uint8(255 * resized_heat)
                colored = cv2.applyColorMap(uint8_heat, cv2.COLORMAP_JET)
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                rgb_arr = np.array(pil_img)
                gradcam_overlay = np.uint8(0.4 * colored + 0.6 * rgb_arr)
            except Exception as e:
                logger.warning("Failed to compute Grad-CAM overlay: %s", e)

        return {
            "status": "success",
            "predicted_breed": top_predictions[0]["breed"],
            "top_predictions": top_predictions,
            "confidence": {
                "level": conf_report.level,
                "score": conf_report.score,
                "is_reliable": conf_report.is_reliable,
                "warning": conf_report.warning,
                "explanation": conf_report.explanation,
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "model_version": f"{self.model_name}-v1.0",
            "gradcam_overlay": gradcam_overlay,
        }
