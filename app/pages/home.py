"""
CanineVision AI — Home Dashboard Page.
"""

import streamlit as st

from app.components.cards import render_badge, render_metric_card


def show():
    # Hero Header
    st.markdown(
        f"""
        <div style="padding: 24px 0 32px 0;">
            <div style="margin-bottom: 12px;">
                {render_badge("ENTERPRISE COMPUTER VISION PLATFORM", "#60a5fa", "rgba(96, 165, 250, 0.15)")}
                {render_badge("120 CANINE SPECIES", "#34d399", "rgba(52, 211, 153, 0.15)")}
                {render_badge("GRAD-CAM EXPLAINABLE AI", "#fbbf24", "rgba(251, 191, 36, 0.15)")}
            </div>
            <h1 style="font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; color: #ffffff;">
                CanineVision <span style="background: linear-gradient(90deg, #3b82f6, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI</span>
            </h1>
            <p style="font-size: 1.2rem; color: #94a3b8; margin-top: 8px; max-width: 800px; line-height: 1.5;">
                Discover the Breed. Understand the Intelligence. An industrial-grade deep learning system for fine-grained canine classification with visual explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Dataset Classes", "120", "Distinct dog breeds", icon="🐕")
    with c2:
        render_metric_card("Curated Images", "20,580", "Stratified 70/15/15", icon="🖼️")
    with c3:
        render_metric_card("Active Architecture", "MobileNetV2", "Transfer Learning", icon="⚡")
    with c4:
        render_metric_card("Inference Latency", "< 60 ms", "CPU Optimized", icon="⏱️")

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 30px 0;'>", unsafe_allow_html=True)

    # Architectural Overview Cards
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px;">
                <h3 style="color: #60a5fa; margin-top: 0; font-size: 1.2rem;">🔬 High-Capacity Transfer Learning</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                    Built on MobileNetV2 with inverted residual bottlenecks and depthwise separable convolutions.
                    Pre-trained on 1.4M ImageNet photographs and fine-tuned for fine-grained morphological discrimination
                    between closely related breeds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px;">
                <h3 style="color: #34d399; margin-top: 0; font-size: 1.2rem;">🔍 Explainable AI (Grad-CAM)</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                    Inspect the model's inner decision-making. Gradient-weighted Class Activation Mapping computes
                    spatial heatmaps directly from the final convolutional feature maps, visually revealing ear, muzzle,
                    and coat attention regions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.info("💡 **Quick Navigation:** Use the left sidebar to access **Breed Prediction**, **Model Analytics**, and **Model Card**.")
