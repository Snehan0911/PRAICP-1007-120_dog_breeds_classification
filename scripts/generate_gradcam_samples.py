"""
Grad-CAM Sample Visualization Generator.
Project: Dog Breeds Classification (PRAICP-1007-120)

Runs Grad-CAM explainability on sample dog photographs and saves
visual explanation side-by-side comparison charts to outputs/gradcam/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocessing import load_and_validate_image
from src.inference.gradcam import GradCAMExplainer


def generate_gradcam_gallery() -> None:
    output_dir = Path("outputs/gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path("dog_photos/samples")
    sample_images = sorted(samples_dir.glob("*.jpg"))
    if not sample_images:
        print("No sample images found in dog_photos/samples!")
        return

    # Load class mapping
    with open("model/class_mapping/class_indices.json", "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
    idx_to_breed = {v: k for k, v in class_mapping.items()}

    explainer = GradCAMExplainer("model/saved_models/best_model.keras")

    print(f"Generating Grad-CAM visualizations for {len(sample_images)} samples...")

    for img_path in sample_images:
        result = explainer.explain_image(img_path)
        pred_idx = result["predicted_class_id"]
        pred_breed = idx_to_breed.get(pred_idx, f"Class {pred_idx}")
        confidence = result["confidence"]
        overlay = result["overlay"]
        raw_img = load_and_validate_image(img_path, target_size=(224, 224))

        # Create dual-panel comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

        ax1.imshow(raw_img)
        ax1.set_title(f"Input: {img_path.stem.replace('_', ' ')}", fontsize=12, fontweight="bold")
        ax1.axis("off")

        ax2.imshow(overlay)
        ax2.set_title(
            f"Grad-CAM Attention Map\nPredicted: {pred_breed} ({confidence:.1%})",
            fontsize=12,
            fontweight="bold",
            color="#1b4f72",
        )
        ax2.axis("off")

        plt.suptitle("Dog Breeds Classification — Visual Explainability (Grad-CAM)", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_file = output_dir / f"gradcam_{img_path.stem}.png"
        fig.savefig(out_file)
        plt.close(fig)
        print(f"  Saved Grad-CAM visualization to: {out_file}")

    print("All Grad-CAM visualizations generated successfully.")


if __name__ == "__main__":
    generate_gradcam_gallery()
