"""
Full Model Evaluation Runner.
Project: Dog Breeds Classification (PRAICP-1007-120)

Runs model evaluation on untouched test set, generates confusion matrices,
performs error analysis on misclassifications, and plots model comparison charts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.error_analysis import perform_error_analysis
from src.evaluation.metrics import evaluate_model_on_test_set
from src.evaluation.model_comparison import generate_model_comparison_chart


def run_full_evaluation(max_samples: int | None = None) -> None:
    print("=" * 70)
    print("  DOG BREEDS CLASSIFICATION — MODEL EVALUATION PIPELINE")
    print("  Project ID: PRAICP-1007-120")
    print("=" * 70)

    # 1. Evaluate Model on Test Set
    print("\n[Step 1/3] Running inference on untouched test partition...")
    eval_data = evaluate_model_on_test_set(max_test_samples=max_samples)

    metrics = eval_data["metrics"]
    y_true = eval_data["y_true"]
    y_pred = eval_data["y_pred"]
    class_names = eval_data["class_names"]
    cls_report = eval_data["classification_report"]

    # 2. Confusion Matrices
    print("\n[Step 2/3] Computing and plotting 120-class confusion matrices...")
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=class_names)

    # 3. Error Analysis & Model Comparison
    print("\n[Step 3/3] Performing error analysis and generating model comparison...")
    perform_error_analysis(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        metrics_summary=metrics,
        classification_report_dict=cls_report,
    )
    generate_model_comparison_chart()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE:")
    print(f"  • Top-1 Accuracy:     {metrics['top_1_accuracy']:.2%}")
    print(f"  • Top-3 Accuracy:     {metrics['top_3_accuracy']:.2%}")
    print(f"  • Macro Precision:    {metrics['precision_macro']:.2%}")
    print(f"  • Macro Recall:       {metrics['recall_macro']:.2%}")
    print(f"  • Macro F1-Score:     {metrics['f1_score_macro']:.2%}")
    print(f"  • Weighted F1-Score:  {metrics['f1_score_weighted']:.2%}")
    print("\nSAVED EVALUATION ARTIFACTS (outputs/evaluation/):")
    print("  1. outputs/evaluation/evaluation_metrics.json")
    print("  2. outputs/evaluation/classification_report.json")
    print("  3. outputs/evaluation/confusion_matrix_normalized.png")
    print("  4. outputs/evaluation/confusion_matrix_top20.png")
    print("  5. outputs/evaluation/error_analysis.json")
    print("  6. outputs/evaluation/model_comparison.png")
    print("  7. docs/evaluation_report.md")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Dog Breeds Classifier.")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max test samples for evaluation"
    )
    args = parser.parse_args()
    run_full_evaluation(max_samples=args.max_samples)
