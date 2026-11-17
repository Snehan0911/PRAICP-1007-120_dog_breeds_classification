"""
Experiment Tracker and Telemetry Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Tracks hyperparameters, training execution history, hardware context,
and generates loss and accuracy curves for model reporting.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ExperimentTracker")


class ExperimentTracker:
    """
    Manages experiment metadata serialization, training metrics persistence,
    and visualization curves.
    """

    def __init__(
        self,
        metadata_dir: str | Path = "model/metadata",
        training_log_dir: str | Path = "log/training",
        figures_dir: str | Path = "outputs/training",
    ) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.training_log_dir = Path(training_log_dir)
        self.training_log_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        model_name: str,
        config: dict[str, Any],
        history: dict[str, list[float]],
        duration_seconds: float,
    ) -> Path:
        """
        Persists full experiment run metadata and training curves.

        Args:
            model_name: Model identifier string.
            config: Hyperparameter dictionary.
            history: Dictionary of metric lists (loss, val_loss, accuracy, val_accuracy, etc.).
            duration_seconds: Wall-clock execution time in seconds.

        Returns:
            Path: Path to serialized metadata JSON file.
        """
        run_record = {
            "project_name": "Dog Breeds Classification",
            "project_id": "PRAICP-1007-120",
            "model_name": model_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "configuration": config,
            "epochs_completed": len(history.get("loss", [])),
            "final_train_loss": round(float(history["loss"][-1]), 4) if history.get("loss") else None,
            "final_val_loss": round(float(history["val_loss"][-1]), 4) if history.get("val_loss") else None,
            "final_train_accuracy": round(float(history["accuracy"][-1]), 4) if history.get("accuracy") else None,
            "final_val_accuracy": round(float(history["val_accuracy"][-1]), 4) if history.get("val_accuracy") else None,
            "best_val_accuracy": round(float(max(history["val_accuracy"])), 4) if history.get("val_accuracy") else None,
            "history": {k: [round(float(v), 5) for v in vals] for k, vals in history.items()},
        }

        def _json_default(o: Any) -> Any:
            if isinstance(o, (np.integer, int)):
                return int(o)
            elif isinstance(o, (np.floating, float)):
                return float(o)
            elif isinstance(o, (np.ndarray, list)):
                return [float(x) for x in o]
            return str(o)

        # Save metadata to model/metadata/
        meta_file = self.metadata_dir / f"experiment_{model_name}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, default=_json_default)

        # Also save standard history.json in log/training/
        hist_file = self.training_log_dir / "history.json"
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, default=_json_default)

        logger.info(f"Experiment metadata serialized to {meta_file} and {hist_file}")

        # Render training curves
        self.plot_training_curves(model_name, history)
        return meta_file

    def plot_training_curves(
        self,
        model_name: str,
        history: dict[str, list[float]],
    ) -> Path:
        """Renders and saves dual-panel loss and accuracy convergence curves."""
        epochs = range(1, len(history.get("loss", [])) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

        # Panel 1: Loss
        if "loss" in history:
            ax1.plot(epochs, history["loss"], "o-", color="#1f77b4", label="Training Loss", linewidth=2)
        if "val_loss" in history:
            ax1.plot(epochs, history["val_loss"], "s--", color="#d62728", label="Validation Loss", linewidth=2)
        ax1.set_title(f"{model_name} — Loss Convergence", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Categorical Cross-Entropy Loss", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=11)

        # Panel 2: Accuracy
        if "accuracy" in history:
            ax2.plot(epochs, history["accuracy"], "o-", color="#2ca02c", label="Training Accuracy", linewidth=2)
        if "val_accuracy" in history:
            ax2.plot(epochs, history["val_accuracy"], "s--", color="#ff7f0e", label="Validation Accuracy", linewidth=2)
        if "val_top_3_accuracy" in history:
            ax2.plot(epochs, history["val_top_3_accuracy"], "^:", color="#9467bd", label="Val Top-3 Accuracy", linewidth=1.5)

        ax2.set_title(f"{model_name} — Accuracy Curves", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(fontsize=11)

        plt.suptitle(f"Dog Breeds Classification — {model_name} Training Performance", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        curve_path = self.figures_dir / f"{model_name}_training_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        logger.info(f"Saved convergence curves to {curve_path}")
        return curve_path
