"""
Unit tests for evaluation metrics and confusion matrix generation.
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

from pathlib import Path
import numpy as np

from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.error_analysis import perform_error_analysis
from src.evaluation.model_comparison import generate_model_comparison_chart


def test_confusion_matrix_plotting(tmp_path: Path):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])
    class_names = ["Chihuahua", "Japanese Spaniel", "Maltese Dog"]

    norm_path, focused_path = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_dir=tmp_path,
    )
    assert norm_path.exists()
    assert focused_path.exists()


def test_error_analysis_execution(tmp_path: Path):
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 2, 2, 1])
    class_names = ["Chihuahua", "Japanese Spaniel", "Maltese Dog"]
    metrics_summary = {
        "top_1_accuracy": 0.50,
        "top_3_accuracy": 1.0,
        "precision_macro": 0.50,
        "recall_macro": 0.50,
        "f1_score_macro": 0.50,
        "f1_score_weighted": 0.50,
        "total_test_samples": 4,
    }
    cls_report = {
        "Chihuahua": {"precision": 1.0, "recall": 0.5, "f1-score": 0.67, "support": 2},
        "Japanese Spaniel": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 1},
        "Maltese Dog": {"precision": 0.5, "recall": 1.0, "f1-score": 0.67, "support": 1},
    }

    doc_file = tmp_path / "test_report.md"
    analysis_data = perform_error_analysis(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        metrics_summary=metrics_summary,
        classification_report_dict=cls_report,
        output_dir=tmp_path,
        docs_path=doc_file,
    )
    assert "top_confused_pairs" in analysis_data
    assert doc_file.exists()


def test_model_comparison_chart(tmp_path: Path):
    chart_path = generate_model_comparison_chart(output_dir=tmp_path)
    assert chart_path.exists()
