"""
CanineVision AI — Main Streamlit Application Entry Point.
"""

import streamlit as st

from app.pages import about, analytics, experiment_tracking, home, model_information, prediction

# Configure page layout and metadata (must be first Streamlit command)
st.set_page_config(
    page_title="CanineVision AI — Dog Breed Intelligence",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme & Enterprise Styling
st.markdown(
    """
    <style>
        /* Base Styling */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #080c14;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Button Styling */
        .stButton>button {
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 8px 16px;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1d4ed8, #2563eb);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        /* Card Containers */
        div[data-testid="stVerticalBlock"] > div[style*="background"] {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 0 20px 0;">
            <span style="font-size: 2rem;">🐕</span>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">CanineVision <span style="color: #3b82f6;">AI</span></div>
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">v1.0.0 &bull; Enterprise Edition</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🎯 Breed Prediction",
            "📊 Model Analytics",
            "🧪 Experiment Tracking",
            "📄 Model Information",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color: #64748b; font-size: 0.75rem; line-height: 1.6;">
            <strong>System:</strong> MobileNetV2<br>
            <strong>Classes:</strong> 120 Breeds<br>
            <strong>Status:</strong> <span style="color: #10b981;">Online &bull; Healthy</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Page Dispatcher
if page == "🏠 Home Dashboard":
    home.show()
elif page == "🎯 Breed Prediction":
    prediction.show()
elif page == "📊 Model Analytics":
    analytics.show()
elif page == "🧪 Experiment Tracking":
    experiment_tracking.show()
elif page == "📄 Model Information":
    model_information.show()
elif page == "ℹ️ About Project":
    about.show()
