"""
CanineVision AI — Official Model Card & Documentation.
"""

import streamlit as st


def show():
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff;">Model Card: CanineVision-MobileNetV2</h2>
            <p style="color: #94a3b8; margin-top: 4px;">Standardized AI model card detailing specifications, intended use, limitations, and performance metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["Overview & Specs", "Preprocessing & Inputs", "Evaluation Envelope", "Limitations & Safety"])

    with t1:
        st.markdown(
            """
            #### 1. Model Details
            - **Model Name:** CanineVision-MobileNetV2-120
            - **Version:** 1.0.0 (Production Release)
            - **Architecture:** MobileNetV2 Backbone + Custom Classification Head
            - **Parameters:** 3.5 Million (3.2M Backbone + 300K Head)
            - **Framework:** TensorFlow 2.21 / Keras 3.14
            - **License:** MIT License
            - **Intended Use:** Automated classification and morphological analysis of canine breeds from visual imagery.
            """
        )

    with t2:
        st.markdown(
            """
            #### 2. Input Specifications & Normalization
            - **Input Tensor Dimensions:** `(None, 224, 224, 3)`
            - **Color Space:** Standard RGB
            - **Normalization Protocol:** Scaled to range `[-1.0, 1.0]` via `tf.keras.applications.mobilenet_v2.preprocess_input`.
            - **Minimum Valid Dimension:** 32x32 pixels
            - **Maximum Supported Payload:** 10 MB per image
            """
        )

    with t3:
        st.markdown(
            """
            #### 3. Evaluation Envelope
            - **Test Dataset:** 3,087 untouched holdout images (15% stratified split).
            - **Top-1 Accuracy:** 84.2%
            - **Top-3 Accuracy:** 96.1%
            - **Macro F1-Score:** 0.835
            - **Average Latency:** 48 ms (Intel 12-core CPU)
            """
        )

    with t4:
        st.markdown(
            """
            #### 4. Failure Modes & Operational Boundaries
            - **Closed-World Constraint:** Model evaluates all inputs within the probability simplex of its 120 recognized classes. It cannot identify wild animals outside this ontology without triggering the uncertainty warning.
            - **Occlusion & Lighting Sensitivity:** Extreme backlight, heavy motion blur, or partial occlusion (<30% of dog visible) reduces Top-1 confidence.
            - **Grad-CAM Interpretation:** Grad-CAM illustrates correlational attention, not biological proof of breed pedigree.
            """
        )
