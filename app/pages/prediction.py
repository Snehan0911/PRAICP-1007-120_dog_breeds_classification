"""
CanineVision AI — Interactive Prediction & Explainability Studio.
"""

from pathlib import Path

import streamlit as st

from app.components.cards import render_badge
from app.components.charts import plot_top_predictions_bar
from src.inference.predictor import BreedPredictor


@st.cache_resource
def get_cached_predictor():
    """Initializes and caches the BreedPredictor across user sessions."""
    return BreedPredictor(model_name="mobilenet_v2")


def show():
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff;">Canine Recognition & Explainability Studio</h2>
            <p style="color: #94a3b8; margin-top: 4px;">Upload any canine photograph to evaluate model prediction, candidate rankings, and Grad-CAM visual attention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = get_cached_predictor()

    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("#### 1. Input Image")
        uploaded_file = st.file_uploader(
            "Choose a dog photograph...",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-resolution, well-lit photos yield highest classification accuracy.",
        )

        sample_choice = st.selectbox(
            "Or select a sample from the dataset:",
            ["None", "Golden Retriever", "Chihuahua", "Siberian Husky", "Samoyed", "Rottweiler"],
        )

        image_to_process = None

        if uploaded_file is not None:
            image_to_process = uploaded_file.read()
            st.image(image_to_process, caption="Uploaded Image", use_container_width=True)
        elif sample_choice != "None":
            # Find a real image from dataset
            breed_keyword = sample_choice.lower().replace(" ", "_")
            dataset_root = Path("dog-breed-120/Dogs class")
            if dataset_root.exists():
                for d in dataset_root.iterdir():
                    if breed_keyword in d.name.lower():
                        imgs = list(d.glob("*.jpg"))
                        if imgs:
                            image_to_process = str(imgs[0])
                            st.image(image_to_process, caption=f"Dataset Sample: {sample_choice}", use_container_width=True)
                            break

        explain_checkbox = st.checkbox("Generate Grad-CAM Attention Heatmap", value=True)

    with col_result:
        st.markdown("#### 2. Model Prediction & Explainability")

        if image_to_process is not None:
            with st.spinner("Executing neural inference & computing gradients..."):
                result = predictor.predict(
                    image_to_process,
                    top_k=3,
                    return_gradcam=explain_checkbox,
                )

            if result.get("status") == "success":
                top_breed = result["predicted_breed"]
                top_prob = result["top_predictions"][0]["percentage"]
                latency = result["inference_time_ms"]
                conf = result["confidence"]

                # Top-1 Hero Card
                badge_color = "#10b981" if conf["level"] == "HIGH" else "#f59e0b" if conf["level"] == "MEDIUM" else "#ef4444"
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.05) 100%);
                                border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #93c5fd; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Primary Prediction</span>
                            {render_badge(f"Confidence: {conf['level']}", badge_color, f"{badge_color}25")}
                        </div>
                        <h2 style="color: #ffffff; margin: 8px 0 4px 0; font-size: 1.8rem; font-weight: 800;">{top_breed}</h2>
                        <div style="color: #cbd5e1; font-size: 0.95rem;">
                            Softmax Probability: <strong style="color: #60a5fa;">{top_prob}</strong> &nbsp;|&nbsp; Latency: <strong style="color: #60a5fa;">{latency} ms</strong>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;">
                            {conf['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if conf["warning"]:
                    st.warning(f"⚠️ **Attention:** {conf['warning']}")

                # Top-3 Probabilities Chart
                st.markdown("##### Top-3 Candidate Probabilities")
                fig = plot_top_predictions_bar(result["top_predictions"])
                st.plotly_chart(fig, use_container_width=True)

                # Grad-CAM Visualization
                if explain_checkbox and result.get("gradcam_overlay") is not None:
                    st.markdown("##### Grad-CAM Visual Attention Heatmap")
                    st.image(
                        result["gradcam_overlay"],
                        caption=f"Grad-CAM overlay highlighting regions salient to '{top_breed}' prediction",
                        use_container_width=True,
                    )
            else:
                st.error(f"Prediction Error: {result.get('message')}")
        else:
            st.info("👈 Upload an image or select a sample from the left panel to trigger real-time inference.")
