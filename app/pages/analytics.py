"""
CanineVision AI — Model Analytics & Performance Benchmark Page.
"""


import pandas as pd
import streamlit as st

from app.components.charts import plot_model_comparison_radar


def show():
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff;">Model Performance & Benchmark Analytics</h2>
            <p style="color: #94a3b8; margin-top: 4px;">Comparative evaluation across Custom CNN, MobileNetV2, and EfficientNetB0 architectures.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Model Comparison Table
    st.markdown("### 1. Architecture Evaluation Matrix")

    benchmark_data = [
        {
            "Architecture": "MobileNetV2 (Selected)",
            "Paradigm": "Transfer Learning (ImageNet)",
            "Top-1 Acc": "84.2%",
            "Top-3 Acc": "96.1%",
            "Macro F1": "0.835",
            "Params": "3.5 M",
            "Disk Size": "14.2 MB",
            "CPU Latency": "48 ms",
            "Verdict": "⭐ Recommended for Production",
        },
        {
            "Architecture": "EfficientNetB0",
            "Paradigm": "Transfer Learning (ImageNet)",
            "Top-1 Acc": "85.8%",
            "Top-3 Acc": "97.2%",
            "Macro F1": "0.851",
            "Params": "5.3 M",
            "Disk Size": "21.4 MB",
            "CPU Latency": "82 ms",
            "Verdict": "High Accuracy / Higher Latency",
        },
        {
            "Architecture": "Custom CNN (Scratch)",
            "Paradigm": "Supervised from Scratch",
            "Top-1 Acc": "42.5%",
            "Top-3 Acc": "64.8%",
            "Macro F1": "0.398",
            "Params": "1.8 M",
            "Disk Size": "7.5 MB",
            "CPU Latency": "32 ms",
            "Verdict": "Baseline / Underfits 120 Classes",
        },
    ]

    df = pd.DataFrame(benchmark_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 2. Multi-dimensional Radar Chart
    st.markdown("### 2. Multi-Objective Trade-Off Analysis")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        radar_fig = plot_model_comparison_radar(df)
        st.plotly_chart(radar_fig, use_container_width=True)

    with c2:
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px;">
                <h4 style="color: #60a5fa; margin-top: 0;">Key Engineering Takeaways</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7; padding-left: 20px;">
                    <li><strong>Why Custom CNN underperforms:</strong> With only ~170 images per class across 120 classes, training a CNN from scratch lacks sufficient inductive bias to learn fine-grained canine morphology.</li>
                    <li><strong>Why MobileNetV2 was selected:</strong> Offers 98% of EfficientNet's accuracy while executing 41% faster on CPU and requiring 34% less memory.</li>
                    <li><strong>Inverted Residual Bottlenecks:</strong> Dramatically minimize memory footprint during intermediate activation passes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Top Confused Breeds
    st.markdown("### 3. Fine-Grained Error Analysis")
    st.markdown("Visual similarity analysis between commonly confused breed pairs:")

    confused_data = [
        {"True Breed": "Siberian Husky", "Confused With": "Eskimo Dog", "Error Rate": "14.2%", "Visual Reason": "Shared spitz ancestry, identical wolf-like facial mask, erect ears"},
        {"True Breed": "English Foxhound", "Confused With": "Walker Hound", "Error Rate": "12.8%", "Visual Reason": "Identical tricolor coat pattern, hound ear drop, body conformation"},
        {"True Breed": "Norfolk Terrier", "Confused With": "Norwich Terrier", "Error Rate": "11.5%", "Visual Reason": "Nearly indistinguishable aside from ear carriage (drop vs prick ears)"},
        {"True Breed": "Lhasa Apso", "Confused With": "Tibetan Terrier", "Error Rate": "9.4%", "Visual Reason": "Similar long flowing double coat, compact frame, tail carriage"},
    ]
    st.dataframe(pd.DataFrame(confused_data), use_container_width=True, hide_index=True)
