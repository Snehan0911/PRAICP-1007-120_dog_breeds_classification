"""
CanineVision AI — MLOps Experiment Tracking with MLflow.

Logs parameters, training dynamics, evaluation metrics, and model metadata
into local MLflow tracking store (mlruns).
"""

import json
from pathlib import Path
from typing import Any

import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    MLflow Experiment Tracker wrapper for deep learning pipelines.

    Args:
        experiment_name: Name of MLflow experiment group.
        tracking_uri: Directory path or URL for MLflow tracking.
    """

    def __init__(
        self,
        experiment_name: str = "CanineVision-Dog-Breed-Classification",
        tracking_uri: str | Path = "mlruns",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = str(Path(tracking_uri).resolve())
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info("MLflow initialized. Tracking URI: %s", self.tracking_uri)

    def start_run(self, run_name: str, tags: dict[str, str] | None = None):
        """Context manager or method to start a new MLflow run."""
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs hyperparameters to active MLflow run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs scalar evaluation/training metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        """Logs an artifact file (e.g. confusion matrix plot, model weights)."""
        p = Path(local_path)
        if p.exists():
            mlflow.log_artifact(str(p))
            logger.info("Logged artifact to MLflow: %s", p.name)
        else:
            logger.warning("Artifact does not exist, cannot log: %s", local_path)

    def save_model_metadata(
        self,
        model_name: str,
        version: str,
        architecture: str,
        metrics: dict[str, float],
        preprocessing_type: str,
        output_dir: str | Path = "artifacts/metadata",
    ) -> Path:
        """
        Saves standardized model metadata JSON.

        Args:
            model_name: Identifier name.
            version: Model version tag.
            architecture: Underlying network backbone.
            metrics: Dict of key evaluation metrics.
            preprocessing_type: Normalization/preprocessing identifier.
            output_dir: Target output directory.

        Returns:
            Path to saved JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_file = output_dir / f"{model_name}_metadata.json"

        metadata = {
            "model_name": model_name,
            "version": version,
            "architecture": architecture,
            "input_shape": [224, 224, 3],
            "num_classes": 120,
            "preprocessing": preprocessing_type,
            "class_mapping_path": "artifacts/class_mapping/class_indices.json",
            "metrics": metrics,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved model metadata: %s", meta_file)
        return meta_file
