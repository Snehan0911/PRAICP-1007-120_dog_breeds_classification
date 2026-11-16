"""
CanineVision AI — Model Architecture Comparison Framework.

Consolidates and compares all three candidate models across:
- Top-1 and Top-3 Accuracy
- Macro and Weighted F1-Score
- Parameter count (Total, Trainable, Non-trainable)
- Model disk size (MB)
- Inference Latency per image (ms)
"""

from pathlib import Path

import pandas as pd


def generate_comparison_table(models_eval_results: list[dict]) -> pd.DataFrame:
    """
    Transforms list of model evaluation summaries into a unified comparison table.

    Args:
        models_eval_results: List of dicts, each containing:
            - model_name
            - top_1_accuracy
            - top_3_accuracy
            - f1_macro
            - f1_weighted
            - param_count_m
            - latency_ms

    Returns:
        Formatted pandas DataFrame.
    """
    df = pd.DataFrame(models_eval_results)
    return df


def save_comparison_report(df: pd.DataFrame, output_path: str | Path) -> None:
    """Saves comparison table to CSV and markdown."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p.with_suffix(".csv"), index=False)
    with open(p.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write("# CanineVision AI — Model Comparison Report\n\n")
        f.write(df.to_markdown(index=False))
