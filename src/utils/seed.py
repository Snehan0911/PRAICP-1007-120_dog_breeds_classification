"""
CanineVision AI — Reproducibility Seed Configuration.

Why this matters:
    Deep learning has randomness built in everywhere:
    - Random weight initialisation
    - Random data shuffling
    - Random dropout during training
    - Random data augmentation

    If you don't fix a seed, every training run produces slightly different
    results, making it impossible to compare experiments fairly or reproduce
    a reported result.

    By calling set_all_seeds(42) at the start of any script or notebook,
    you ensure the SAME sequence of random numbers is used every time.

    Note: Full reproducibility on CPU is achievable. On GPU it is harder
    due to non-deterministic floating point operations in CUDA kernels.
    Since we are on CPU, this gives us reliable reproducibility.

Usage:
    from src.utils.seed import set_all_seeds
    set_all_seeds(42)
"""

import os
import random

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure
    reproducible results across runs.

    Args:
        seed: Integer seed value. 42 is used by convention.
              Any fixed integer produces reproducible results.
    """
    # 1. Python's built-in random module (used by some data augmentation)
    random.seed(seed)

    # 2. Python hash seed — affects dictionary ordering and set iteration
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy — affects random array operations, shuffling
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy not available — NumPy seed not set")

    # 4. TensorFlow — affects weight initialisation and dropout
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not available — TF seed not set")

    logger.info("All random seeds set to: %d", seed)
