"""
Model Evaluation Metrics Pipeline.
Project: Dog Breeds Classification (PRAICP-1007-120)

Evaluates trained deep neural network on the untouched test dataset partition,
computing Top-1 Accuracy, Top-3 Accuracy, Precision, Recall, and Macro/Weighted F1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.data.dataset_loader import DatasetLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ModelEvaluator")


def evaluate_model_on_test_set(
    model_path: str | Path = "model/saved_models/best_model.keras",
    batch_size: int = 32,
    max_test_samples: int | None = None,
    output_dir: str | Path = "outputs/evaluation",
    metadata_dir: str | Path = "model/metadata",
) -> dict[str, Any]:
    """
    Runs model inference on untouched test set and computes classification metrics.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta_path = Path(metadata_dir)
    meta_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trained model from: {model_file}")
    model = tf.keras.models.load_model(str(model_file), compile=False)

    loader = DatasetLoader()
    test_df = loader.get_split_dataframe("test")
    if max_test_samples:
        test_df = test_df.iloc[:max_test_samples].copy()

    test_ds = loader.get_dataset(
        split="test",
        batch_size=batch_size,
        augment=False,
    )

    logger.info(f"Running predictions on {len(test_df)} untouched test samples...")
    y_true_list: list[int] = []
    y_pred_probs_list: list[np.ndarray] = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred_probs_list.append(preds)
        if max_test_samples and len(y_true_list) >= max_test_samples:
            break

    y_true = np.array(
        y_true_list[:max_test_samples] if max_test_samples else y_true_list
    )
    y_pred_probs = np.vstack(y_pred_probs_list)[: len(y_true)]

    y_pred = np.argmax(y_pred_probs, axis=1)

    # Top-1 Accuracy
    top_1_accuracy = float(np.mean(y_true == y_pred))

    # Top-3 Accuracy
    top_3_indices = np.argsort(y_pred_probs, axis=1)[:, -3:]
    top_3_accuracy = float(
        np.mean([true_class in top3 for true_class, top3 in zip(y_true, top_3_indices)])
    )

    # Precision, Recall, F1 (Macro and Weighted)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Class report
    class_names = [loader.idx_to_class[i] for i in range(loader.num_classes)]
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(loader.num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics_results = {
        "model_path": str(model_file),
        "total_test_samples": len(y_true),
        "total_classes": int(loader.num_classes),
        "top_1_accuracy": round(float(top_1_accuracy), 4),
        "top_3_accuracy": round(float(top_3_accuracy), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_score_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weighted), 4),
        "recall_weighted": round(float(r_weighted), 4),
        "f1_score_weighted": round(float(f1_weighted), 4),
    }

    # Save metrics JSON
    metrics_file = out_path / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=2)

    meta_eval_file = meta_path / "evaluation_results.json"
    with open(meta_eval_file, "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=2)

    report_file = out_path / "classification_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(cls_report, f, indent=2)

    logger.info(
        f"Evaluation complete on {len(y_true)} test samples: "
        f"Top-1 Acc: {top_1_accuracy:.4f}, Top-3 Acc: {top_3_accuracy:.4f}, "
        f"Macro F1: {f1_macro:.4f}, Weighted F1: {f1_weighted:.4f}"
    )

    return {
        "metrics": metrics_results,
        "classification_report": cls_report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
        "class_names": class_names,
    }


if __name__ == "__main__":
    results = evaluate_model_on_test_set()
    print("\nEVALUATION RESULTS:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")
