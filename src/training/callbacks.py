"""
Training Callbacks Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Configures production training callbacks: ModelCheckpoint, EarlyStopping,
ReduceLROnPlateau, CSVLogger, and TensorBoard logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("TrainingCallbacks")


def get_training_callbacks(
    model_name: str = "mobilenet_v2",
    checkpoint_dir: str | Path = "model/checkpoints",
    training_log_dir: str | Path = "log/training",
    tensorboard_log_dir: str | Path = "log/tensorboard",
    patience_early_stopping: int = 5,
    patience_reduce_lr: int = 2,
    reduce_lr_factor: float = 0.5,
    min_learning_rate: float = 1e-6,
) -> list[tf.keras.callbacks.Callback]:
    """
    Constructs a suite of training callbacks for reliable convergence and telemetry.

    Callbacks Included:
      1. ModelCheckpoint: Saves best weights monitoring 'val_accuracy'.
      2. EarlyStopping: Halts training if 'val_loss' does not improve, restoring best weights.
      3. ReduceLROnPlateau: Decays learning rate when validation loss plateaus.
      4. TensorBoard: Writes real-time scalar metrics for dashboard visualization.
      5. CSVLogger: Persists epoch-by-epoch loss and accuracy metrics to disk.

    Args:
        model_name: Identifier for naming checkpoints and log subdirectories.
        checkpoint_dir: Directory where model weights will be saved.
        training_log_dir: Directory where CSV training logs will be written.
        tensorboard_log_dir: Directory where TensorBoard scalar logs will be saved.
        patience_early_stopping: Epochs without improvement before early termination.
        patience_reduce_lr: Epochs without improvement before decaying learning rate.
        reduce_lr_factor: Multiplier for learning rate reduction.
        min_learning_rate: Lower bound for learning rate decay.

    Returns:
        list[tf.keras.callbacks.Callback]: Configured Keras callbacks.
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_log_path = Path(training_log_dir)
    train_log_path.mkdir(parents=True, exist_ok=True)

    tb_log_path = Path(tensorboard_log_dir) / model_name
    tb_log_path.mkdir(parents=True, exist_ok=True)

    # 1. Model Checkpoint (saves best model by validation accuracy)
    checkpoint_file = ckpt_path / f"{model_name}_best.weights.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_file),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    # 2. Early Stopping (prevents overfitting and wasteful CPU cycles)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience_early_stopping,
        restore_best_weights=True,
        verbose=1,
    )

    # 3. Learning Rate Reduction on Plateau
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=reduce_lr_factor,
        patience=patience_reduce_lr,
        min_lr=min_learning_rate,
        verbose=1,
    )

    # 4. TensorBoard Logging
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=str(tb_log_path),
        histogram_freq=0,  # Omit histograms for fast CPU execution
        write_graph=False,
        write_images=False,
        update_freq="epoch",
    )

    # 5. CSV Logger for offline persistence and history plots
    csv_file = train_log_path / f"{model_name}_training_history.csv"
    csv_logger_cb = tf.keras.callbacks.CSVLogger(
        filename=str(csv_file),
        separator=",",
        append=False,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        checkpoint_cb,
        early_stopping_cb,
        reduce_lr_cb,
        tensorboard_cb,
        csv_logger_cb,
    ]
    logger.info(
        f"Configured 5 callbacks for {model_name} (Checkpoint: {checkpoint_file}, TensorBoard: {tb_log_path})"
    )
    return callbacks
