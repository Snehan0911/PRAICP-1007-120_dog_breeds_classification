"""
Model Training Pipeline Orchestrator.
Project: Dog Breeds Classification (PRAICP-1007-120)

Orchestrates reproducible data loading, neural network instantiation,
callback configuration, multi-epoch training, checkpointing, and telemetry.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.data.dataset_loader import DatasetLoader
from src.models.model_factory import create_model
from src.training.callbacks import get_training_callbacks
from src.training.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ModelTrainer")


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Reproducible random seed set to: {seed}")


class ModelTrainer:
    """
    Executes model training with automatic checkpointing, early stopping,
    and experiment telemetry.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        num_classes: int = 120,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        saved_models_dir: str | Path = "model/saved_models",
        checkpoints_dir: str | Path = "model/checkpoints",
        training_log_dir: str | Path = "log/training",
        tensorboard_log_dir: str | Path = "log/tensorboard",
        metadata_dir: str | Path = "model/metadata",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.saved_models_dir = Path(saved_models_dir)
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = Path(checkpoints_dir)
        self.training_log_dir = Path(training_log_dir)
        self.tensorboard_log_dir = Path(tensorboard_log_dir)
        self.metadata_dir = Path(metadata_dir)

        set_seed(self.seed)
        self.loader = DatasetLoader()
        self.tracker = ExperimentTracker(
            metadata_dir=self.metadata_dir,
            training_log_dir=self.training_log_dir,
        )

    def train(
        self,
        steps_per_epoch: int | None = None,
        validation_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Executes end-to-end training and saves best and final models.
        """
        logger.info(
            f"Starting training run for: {self.model_name} (Epochs: {self.epochs}, Batch Size: {self.batch_size})"
        )

        # 1. Prepare Datasets
        train_ds = self.loader.get_dataset(
            split="train",
            model_name=self.model_name,
            batch_size=self.batch_size,
            augment=True,
        )
        val_ds = self.loader.get_dataset(
            split="val",
            model_name=self.model_name,
            batch_size=self.batch_size,
            augment=False,
        )

        # 2. Instantiate Model
        model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
        )

        # 3. Configure Callbacks
        callbacks = get_training_callbacks(
            model_name=self.model_name,
            checkpoint_dir=self.checkpoints_dir,
            training_log_dir=self.training_log_dir,
            tensorboard_log_dir=self.tensorboard_log_dir,
        )

        # 4. Execute Training Loop
        start_time = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,
        )
        duration = time.time() - start_time

        # 5. Save Final Export Model
        best_model_export = self.saved_models_dir / "best_model.keras"
        final_model_export = self.saved_models_dir / f"{self.model_name}_final.keras"

        model.save(str(best_model_export))
        model.save(str(final_model_export))
        logger.info(
            f"Exported production models to: {best_model_export} and {final_model_export}"
        )

        # 6. Log Experiment Metadata
        config_record = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "seed": self.seed,
            "total_params": int(model.count_params()),
            "trainable_params": int(
                sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
            ),
        }
        self.tracker.log_run(
            model_name=self.model_name,
            config=config_record,
            history=history.history,
            duration_seconds=duration,
        )

        logger.info(
            f"Training successfully completed in {duration:.1f}s. "
            f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}"
        )
        return {
            "model": model,
            "history": history.history,
            "duration_seconds": duration,
            "best_model_path": str(best_model_export),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dog Breeds Classifier.")
    parser.add_argument(
        "--model", type=str, default="mobilenet_v2", help="Model architecture"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None, help="Steps per epoch limit"
    )
    parser.add_argument(
        "--val-steps", type=int, default=None, help="Validation steps limit"
    )

    args = parser.parse_args()

    trainer = ModelTrainer(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    trainer.train(
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
    )
