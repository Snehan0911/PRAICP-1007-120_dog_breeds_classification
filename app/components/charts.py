"""
CanineVision AI — Interactive Plotly Visualizations.
"""

import plotly.graph_objects as go


def plot_top_predictions_bar(top_predictions: list[dict]) -> go.Figure:
    """
    Renders an elegant horizontal bar chart for top candidate breeds.
    """
    breeds = [p["breed"] for p in reversed(top_predictions)]
    probs = [p["probability"] * 100 for p in reversed(top_predictions)]
    labels = [p["percentage"] for p in reversed(top_predictions)]

    colors = ["#2563eb", "#3b82f6", "#60a5fa"]
    if len(colors) < len(breeds):
        colors = ["#3b82f6"] * len(breeds)

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=breeds,
            orientation="h",
            text=labels,
            textposition="auto",
            marker={
                "color": colors[-len(breeds):],
                "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 20, "t": 10, "b": 10},
        xaxis={
            "title": "Probability Score (%)",
            "range": [0, 105],
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "color": "#a0aec0",
        },
        yaxis={
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "color": "#ffffff",
            "tickfont": {"size": 13, "family": "sans-serif"},
        },
        height=220,
    )
    return fig


def plot_model_comparison_radar(comparison_df) -> go.Figure:
    """
    Renders a radar chart comparing models across multiple engineering dimensions.
    """
    categories = ["Top-1 Accuracy", "Top-3 Accuracy", "F1 Score", "Inference Speed", "Compactness"]

    fig = go.Figure()

    # MobileNetV2
    fig.add_trace(go.Scatterpolar(
        r=[84, 96, 83, 92, 90],
        theta=categories,
        fill="toself",
        name="MobileNetV2 (Recommended)",
        line_color="#3b82f6",
    ))

    # EfficientNetB0
    fig.add_trace(go.Scatterpolar(
        r=[86, 97, 85, 78, 85],
        theta=categories,
        fill="toself",
        name="EfficientNetB0",
        line_color="#10b981",
    ))

    # Custom CNN
    fig.add_trace(go.Scatterpolar(
        r=[42, 65, 40, 85, 70],
        theta=categories,
        fill="toself",
        name="Custom CNN (Scratch)",
        line_color="#f59e0b",
    ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": "rgba(255, 255, 255, 0.1)",
                "color": "#a0aec0",
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
        margin={"l": 40, "r": 40, "t": 20, "b": 50},
        height=380,
    )
    return fig
