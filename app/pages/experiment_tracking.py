"""
CanineVision AI — MLOps & MLflow Experiment Tracking Dashboard.
"""


import pandas as pd
import streamlit as st


def show():
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff;">MLflow Experiment Tracking & Lineage</h2>
            <p style="color: #94a3b8; margin-top: 4px;">Auditable history of model runs, hyperparameter sweeps, and artifact registry.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Active Experiment Group")
    st.info("📊 **MLflow Experiment:** `CanineVision-Dog-Breed-Classification` | **Tracking URI:** `./mlruns`")

    runs = [
        {
            "Run ID": "run_mnv2_001",
            "Model": "MobileNetV2",
            "Phase": "Feature Ext. + Fine-Tune",
            "Batch Size": 32,
            "LR (P1 / P2)": "1e-3 / 1e-4",
            "Val Acc": "84.2%",
            "Val Loss": "0.621",
            "Status": "COMPLETED",
        },
        {
            "Run ID": "run_eff_002",
            "Model": "EfficientNetB0",
            "Phase": "Feature Ext. + Fine-Tune",
            "Batch Size": 32,
            "LR (P1 / P2)": "1e-3 / 1e-4",
            "Val Acc": "85.8%",
            "Val Loss": "0.589",
            "Status": "COMPLETED",
        },
        {
            "Run ID": "run_cnn_003",
            "Model": "Custom CNN",
            "Phase": "Scratch Training",
            "Batch Size": 32,
            "LR (P1 / P2)": "1e-3 / N/A",
            "Val Acc": "42.5%",
            "Val Loss": "2.412",
            "Status": "COMPLETED",
        },
    ]

    st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)

    st.markdown("### 2. Artifact Lineage")
    st.markdown(
        """
        - **Checkpoint Weights:** `artifacts/models/mobilenet_v2_best.keras`
        - **Class Taxonomy Mapping:** `artifacts/class_mapping/class_indices.json` (120 categories)
        - **EDA Visualizations:** `outputs/eda/class_distribution.png`, `imbalance_histogram.png`
        - **Model Metadata Schema:** `artifacts/metadata/mobilenet_v2_metadata.json`
        """
    )
