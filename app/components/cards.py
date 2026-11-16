"""
CanineVision AI — Custom Glassmorphic Metric Cards for Streamlit.
"""

import streamlit as st


def render_metric_card(title: str, value: str, subtitle: str = "", delta: str | None = None, icon: str = "🐕"):
    """Renders a styled dark glassmorphic metric card."""
    delta_html = f"<span style='color: #4caf50; font-size: 0.85rem; font-weight: 600;'>▲ {delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 12px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: #a0aec0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">{title}</span>
                <span style="font-size: 1.2rem;">{icon}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <span style="color: #ffffff; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em;">{value}</span>
                {delta_html}
            </div>
            <div style="color: #718096; font-size: 0.8rem; margin-top: 4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(text: str, color: str = "#3b82f6", bg: str = "rgba(59, 130, 246, 0.15)"):
    """Renders an inline modern pill badge."""
    return f"""
    <span style="
        background: {bg};
        color: {color};
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid {color}40;
        display: inline-block;
    ">{text}</span>
    """
