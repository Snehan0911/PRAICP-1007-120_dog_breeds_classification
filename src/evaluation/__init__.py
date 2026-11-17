"""
Model evaluation, metrics, confusion matrix, and error analysis.
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

from .confusion_matrix import plot_confusion_matrix
from .error_analysis import perform_error_analysis
from .metrics import evaluate_model_on_test_set

__all__ = [
    "evaluate_model_on_test_set",
    "perform_error_analysis",
    "plot_confusion_matrix",
]
