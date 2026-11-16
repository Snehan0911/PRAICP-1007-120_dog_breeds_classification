"""
CanineVision AI — About Project & Architecture.
"""

import streamlit as st


def show():
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff;">About CanineVision AI</h2>
            <p style="color: #94a3b8; margin-top: 4px;">Enterprise-Grade Canine Intelligence & Computer Vision Platform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Project Identification
        - **Project ID:** PRAICP-1007-120
        - **Original Project Name:** Dog Breeds Classification
        - **System Category:** Deep Learning / Computer Vision / MLOps / Explainable AI
        - **Technology Stack:** Python 3.13, TensorFlow 2.21, Keras 3.14, FastAPI, Streamlit, MLflow, Plotly, Pytest, Docker

        ### System Architecture
        CanineVision AI is engineered as a clean, modular monolith:
        1. **Data Layer (`src/data`):** Automated discovery, validation, stratified splitting (zero-leakage), and high-throughput streaming via `tf.data`.
        2. **Machine Learning Layer (`src/models`):** Custom CNN, MobileNetV2, and EfficientNetB0 architectures instantiated through a unified factory.
        3. **Training & MLOps Layer (`src/training`):** Two-phase transfer learning engine with dynamic learning rate decay, checkpointing, and MLflow experiment logging.
        4. **Explainable AI Layer (`src/inference/gradcam.py`):** Grad-CAM heatmaps visualizing spatial convolutional activations.
        5. **Inference & Validation Layer (`src/inference`):** Byte-level validation, architecture-specific normalisation, and calibrated confidence tiers.
        6. **API Layer (`api`):** Asynchronous REST API with Pydantic schemas, lifespan pre-warming, and OpenAPI docs.
        7. **UI Layer (`app`):** SaaS-grade Streamlit portal with real-time inference, Plotly visualizations, and model card documentation.
        """
    )
