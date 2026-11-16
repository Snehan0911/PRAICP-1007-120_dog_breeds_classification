"""
CanineVision AI — Training Callbacks.

Configures robust callbacks to:
1. Save the best checkpoint based on validation accuracy (ModelCheckpoint).
2. Halt early when validation loss plateaus to conserve compute and avoid overfitting (EarlyStopping).
3. Dynamically decay learning rate when progress stalls (ReduceLROnPlateau).
4. Persist per-epoch epoch loss and metrics to CSV for auditability and dashboard plotting (CSVLogger).
"""

from pathlib import Path

import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_training_callbacks(
    model_name: str,
    checkpoint_dir: str | Path = "artifacts/models",
    history_dir: str | Path = "outputs/training",
    patience_early_stopping: int = 5,
    patience_reduce_lr: int = 3,
    reduce_lr_factor: float = 0.5,
    min_learning_rate: float = 1e-6,
) -> list[tf.keras.callbacks.Callback]:
    """
    Constructs a list of standard production training callbacks.

    Args:
        model_name: Identifier for the model run (e.g. 'mobilenet_v2').
        checkpoint_dir: Directory where the best .keras model file is saved.
        history_dir: Directory where training history CSV is logged.
        patience_early_stopping: Epochs to wait without validation loss improvement.
        patience_reduce_lr: Epochs to wait before decaying LR.
        reduce_lr_factor: Multiplier for learning rate reduction.
        min_learning_rate: Floor for learning rate.

    Returns:
        List of configured tf.keras.callbacks.Callback objects.
    """
    checkpoint_dir = Path(checkpoint_dir)
    history_dir = Path(history_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{model_name}_best.keras"
    csv_log_path = history_dir / f"{model_name}_history.csv"

    callbacks = [
        # 1. ModelCheckpoint: Saves best weights only
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # 2. EarlyStopping: Prevents wasteful training and overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        # 3. ReduceLROnPlateau: Fine-tunes optimization steps as loss flattens
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=min_learning_rate,
            verbose=1,
        ),
        # 4. CSVLogger: Epoch-by-epoch logging
        tf.keras.callbacks.CSVLogger(
            filename=str(csv_log_path),
            separator=",",
            append=False,
        ),
    ]

    logger.info(
        "Configured %d callbacks for %s. Checkpoint target: %s",
        len(callbacks),
        model_name,
        checkpoint_path,
    )
    return callbacks
