"""
Model Comparison and Benchmark Plotting.
Project: Dog Breeds Classification (PRAICP-1007-120)

Generates comparative visualizations evaluating parameters, FLOPs,
and latency profiles across Custom CNN, MobileNetV2, and EfficientNetB0.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ModelComparison")


def generate_model_comparison_chart(
    output_dir: str | Path = "outputs/evaluation",
) -> Path:
    """Generates visual comparison chart across all three candidate models."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models_data = [
        {
            "name": "Custom CNN\n(From Scratch)",
            "params_m": 0.486,
            "macs_m": 250,
            "cpu_ms": 12,
            "accuracy_pot": "Baseline (30-45%)",
            "color": "#1f77b4",
        },
        {
            "name": "MobileNetV2\n(Selected Engine)",
            "params_m": 2.62,
            "macs_m": 300,
            "cpu_ms": 22,
            "accuracy_pot": "High (75-85%)",
            "color": "#2ca02c",
        },
        {
            "name": "EfficientNetB0\n(Compound Scaling)",
            "params_m": 4.36,
            "macs_m": 390,
            "cpu_ms": 38,
            "accuracy_pot": "High (77-87%)",
            "color": "#ff7f0e",
        },
    ]

    names = [m["name"] for m in models_data]
    params = [m["params_m"] for m in models_data]
    latencies = [m["cpu_ms"] for m in models_data]
    colors = [m["color"] for m in models_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # 1. Parameter Count Comparison
    bars1 = ax1.bar(names, params, color=colors, width=0.55, edgecolor="black")
    ax1.set_title(
        "Total Parameters (Millions — Lower is Lighter)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Parameters (M)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.1,
            f"{yval:.2f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. CPU Latency Profile
    bars2 = ax2.bar(names, latencies, color=colors, width=0.55, edgecolor="black")
    ax2.set_title(
        "Average CPU Inference Latency (ms — Lower is Faster)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylabel("Latency (milliseconds)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.8,
            f"{yval} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.suptitle(
        "Dog Breeds Classification — Architectural Benchmark & Trade-off Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    out_file = out_path / "model_comparison.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info(f"Saved model comparison chart to {out_file}")
    return out_file


if __name__ == "__main__":
    generate_model_comparison_chart()
