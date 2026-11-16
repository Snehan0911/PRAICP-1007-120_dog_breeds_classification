"""
CanineVision AI — CLI Single Image Prediction Script.

Usage:
    python scripts/predict_image.py --image path/to/dog.jpg --explain
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import BreedPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CanineVision AI — Predict Breed CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input dog image")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Model architecture")
    parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM explainability")

    args = parser.parse_args()

    predictor = BreedPredictor(model_name=args.model)
    result = predictor.predict(args.image, return_gradcam=args.explain)

    if result.get("status") != "success":
        print(f"Error: {result.get('message')}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  CANINEVISION AI — INFERENCE RESULT")
    print("=" * 60)
    print(f"  Predicted Breed : {result['predicted_breed']}")
    print(f"  Confidence Tier : {result['confidence']['level']} ({result['confidence']['score']*100:.1f}%)")
    print(f"  Inference Time  : {result['inference_time_ms']} ms")
    print(f"  Model Version   : {result['model_version']}")

    if result['confidence']['warning']:
        print(f"  Warning         : {result['confidence']['warning']}")

    print("\n  Top-3 Predictions:")
    for i, p in enumerate(result['top_predictions'], 1):
        print(f"    {i}. {p['breed']:<35} {p['percentage']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
