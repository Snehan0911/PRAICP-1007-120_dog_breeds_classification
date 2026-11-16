"""
CanineVision AI — Production Training Engine.

Orchestrates the entire training lifecycle:
1. Dataset pipeline instantiation.
2. Model compilation and architecture setup.
3. Two-phase transfer learning (Feature Extraction -> Fine Tuning).
4. Callback management (EarlyStopping, Checkpointing, LR Decay, CSV Logging).
5. Experiment tracking via MLflow.
6. Metric logging and visualization generation.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data.pipeline import DataPipelineBuilder
from src.models.model_factory import get_model
from src.training.callbacks import get_training_callbacks
from src.training.experiment_tracker import ExperimentTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Modular, reusable training engine for CanineVision AI models.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        seed: int = 42,
        experiment_name: str = "CanineVision-Dog-Breed-Classification",
    ) -> None:
        self.model_name = model_name.lower()
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.pipeline_builder = DataPipelineBuilder(
            model_name=self.model_name,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
        )
        self.tracker = ExperimentTracker(experiment_name=experiment_name)

    def train(
        self,
        splits_dir: str | Path = "data/splits",
        epochs_phase1: int = 15,
        epochs_phase2: int = 10,
        lr_phase1: float = 0.001,
        lr_phase2: float = 0.0001,
        fine_tune_at: int = -30,
        checkpoint_dir: str | Path = "artifacts/models",
        history_dir: str | Path = "outputs/training",
    ) -> dict[str, Any]:
        """
        Executes end-to-end model training.

        Args:
            splits_dir: Directory containing train.csv, val.csv, test.csv.
            epochs_phase1: Epochs for initial training / feature extraction.
            epochs_phase2: Epochs for fine-tuning unfrozen layers (transfer learning).
            lr_phase1: Phase 1 learning rate.
            lr_phase2: Phase 2 learning rate (typically 10x smaller).
            fine_tune_at: Layer index from which to unfreeze in Phase 2.
            checkpoint_dir: Directory to save best model checkpoint.
            history_dir: Directory to save training plots and logs.

        Returns:
            Dict containing trained model and history logs.
        """
        checkpoint_dir = Path(checkpoint_dir)
        history_dir = Path(history_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading streaming datasets from %s...", splits_dir)
        train_ds, val_ds, test_ds = self.pipeline_builder.load_from_splits(splits_dir)

        run_name = f"{self.model_name}_training_run"
        with self.tracker.start_run(run_name=run_name):
            # Log initial hyperparameters
            self.tracker.log_params({
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "image_size": str(self.image_size),
                "epochs_phase1": epochs_phase1,
                "epochs_phase2": epochs_phase2,
                "lr_phase1": lr_phase1,
                "lr_phase2": lr_phase2,
            })

            # --- Phase 1: Feature Extraction (or scratch training for CNN) ---
            logger.info("=== Starting Phase 1: %s (epochs=%d, lr=%f) ===", self.model_name, epochs_phase1, lr_phase1)
            model = get_model(
                model_name=self.model_name,
                input_shape=(*self.image_size, 3),
                num_classes=120,
                learning_rate=lr_phase1,
                freeze_backbone=True,
            )

            callbacks = get_training_callbacks(
                model_name=self.model_name,
                checkpoint_dir=checkpoint_dir,
                history_dir=history_dir,
            )

            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs_phase1,
                callbacks=callbacks,
            )

            # --- Phase 2: Controlled Fine-Tuning (for MobileNetV2 / EfficientNet) ---
            if epochs_phase2 > 0 and self.model_name != "custom_cnn":
                logger.info("=== Starting Phase 2 Fine-Tuning: unfreezing top layers ===")
                # Access base model inside the functional graph
                base_model = None
                for layer in model.layers:
                    if isinstance(layer, tf.keras.Model):
                        base_model = layer
                        break

                if base_model is not None:
                    base_model.trainable = True
                    # Freeze all layers before fine_tune_at
                    freeze_idx = len(base_model.layers) + fine_tune_at if fine_tune_at < 0 else fine_tune_at
                    for layer in base_model.layers[:freeze_idx]:
                        layer.trainable = False

                    logger.info("Unfroze layers from %d onwards for fine tuning.", freeze_idx)

                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase2),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )

                    total_epochs = epochs_phase1 + epochs_phase2
                    history2 = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=total_epochs,
                        initial_epoch=len(history1.epoch),
                        callbacks=callbacks,
                    )
                else:
                    logger.warning("Could not isolate base model layer for fine-tuning.")

            # Generate and save training convergence plot
            csv_path = history_dir / f"{self.model_name}_history.csv"
            plot_path = history_dir / f"{self.model_name}_training_curves.png"
            if csv_path.exists():
                self._plot_training_curves(csv_path, plot_path)
                self.tracker.log_artifact(plot_path)

            best_model_path = checkpoint_dir / f"{self.model_name}_best.keras"
            if best_model_path.exists():
                self.tracker.log_artifact(best_model_path)

            logger.info("Training cycle completed for %s.", self.model_name)
            return {"model": model, "best_model_path": best_model_path}

    def _plot_training_curves(self, csv_path: Path, output_path: Path) -> None:
        """Generates publication-quality accuracy and loss plots from CSV log."""
        df = pd.read_csv(csv_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax1.set_facecolor("#1a1a2e")
        ax2.set_facecolor("#1a1a2e")

        # Accuracy
        ax1.plot(df["epoch"], df["accuracy"], label="Train Accuracy", color="#4fc3f7", lw=2)
        if "val_accuracy" in df.columns:
            ax1.plot(df["epoch"], df["val_accuracy"], label="Val Accuracy", color="#ff7043", lw=2)
        ax1.set_title("Model Accuracy Convergence", color="#ffffff", fontsize=12)
        ax1.set_xlabel("Epoch", color="#e0e0e0")
        ax1.set_ylabel("Accuracy", color="#e0e0e0")
        ax1.legend(facecolor="#2a2a3e", labelcolor="#ffffff")
        ax1.tick_params(colors="#e0e0e0")

        # Loss
        ax2.plot(df["epoch"], df["loss"], label="Train Loss", color="#4fc3f7", lw=2)
        if "val_loss" in df.columns:
            ax2.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#ff7043", lw=2)
        ax2.set_title("Model Loss Convergence", color="#ffffff", fontsize=12)
        ax2.set_xlabel("Epoch", color="#e0e0e0")
        ax2.set_ylabel("Loss", color="#e0e0e0")
        ax2.legend(facecolor="#2a2a3e", labelcolor="#ffffff")
        ax2.tick_params(colors="#e0e0e0")

        for ax in (ax1, ax2):
            for s in ax.spines.values():
                s.set_visible(False)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        logger.info("Saved convergence curves to: %s", output_path)
