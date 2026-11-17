"""
Dataset Loader Orchestrator.
Project: Dog Breeds Classification (PRAICP-1007-120)

Provides a unified high-level interface for loading training, validation,
and testing data pipelines alongside breed names and class mappings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .data_split import create_stratified_splits
from .pipeline import build_tf_dataset
from .preprocessing import DEFAULT_IMAGE_SIZE


class DatasetLoader:
    """
    High-level manager for dataset splits, class mappings, and tf.data pipelines.
    """

    def __init__(
        self,
        splits_dir: str | Path = "data/splits",
        class_mapping_path: str | Path = "model/class_mapping/class_indices.json",
        labels_csv_path: str | Path = "labels.csv",
    ) -> None:
        self.splits_dir = Path(splits_dir)
        self.class_mapping_path = Path(class_mapping_path)
        self.labels_csv_path = Path(labels_csv_path)

        self._ensure_splits_exist()
        self.class_indices: dict[str, int] = self._load_class_indices()
        self.idx_to_class: dict[int, str] = {
            v: k for k, v in self.class_indices.items()
        }
        self.breed_names: list[str] = [
            self.idx_to_class[i] for i in range(len(self.class_indices))
        ]

    def _ensure_splits_exist(self) -> None:
        train_p = self.splits_dir / "train.csv"
        val_p = self.splits_dir / "val.csv"
        test_p = self.splits_dir / "test.csv"

        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            create_stratified_splits(
                labels_csv_path=self.labels_csv_path,
                output_dir=self.splits_dir,
            )

    def _load_class_indices(self) -> dict[str, int]:
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(
                f"Class mapping file not found at: {self.class_mapping_path}"
            )
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def num_classes(self) -> int:
        return len(self.class_indices)

    def get_split_dataframe(self, split: str = "train") -> pd.DataFrame:
        target = self.splits_dir / f"{split}.csv"
        if not target.exists():
            raise ValueError(
                f"Unknown split: '{split}'. Expected 'train', 'val', or 'test'."
            )
        return pd.read_csv(target)

    def get_dataset(
        self,
        split: str = "train",
        model_name: str = "mobilenet_v2",
        batch_size: int = 32,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = True,
    ) -> tf.data.Dataset:
        """
        Creates a tf.data.Dataset for the requested split.
        """
        df = self.get_split_dataframe(split)
        is_training = split == "train"

        return build_tf_dataset(
            df_or_path=df,
            num_classes=self.num_classes,
            model_name=model_name,
            batch_size=batch_size,
            target_size=target_size,
            is_training=is_training,
            augment=(augment if is_training else False),
        )


if __name__ == "__main__":
    loader = DatasetLoader()
    print(
        f"DatasetLoader initialized successfully. Total Classes: {loader.num_classes}"
    )
    train_ds = loader.get_dataset("train", batch_size=16)
    for images, labels in train_ds.take(1):
        print(
            f"Sample Batch - Images Shape: {images.shape}, Labels Shape: {labels.shape}"
        )
        print(
            f"Images Value Range: min={tf.reduce_min(images):.3f}, max={tf.reduce_max(images):.3f}"
        )
