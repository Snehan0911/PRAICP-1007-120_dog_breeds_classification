"""
Dog Breeds Classification – Flask Web Application.
Project Name: Dog Breeds Classification
Project ID: PRAICP-1007-120

Provides an interactive web interface for uploading canine images, viewing
fine-grained predictions across 120 breeds, exploring Grad-CAM explainability,
and reviewing model evaluation analytics.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.image_validator import ImageValidator
from src.inference.predictor import DogBreedPredictor, format_breed_display_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("WebApp")

# App Initialization
app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "dog-breeds-classification-secret-key-praicp-1007-120")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Paths
UPLOAD_FOLDER = PROJECT_ROOT / "dog_photos" / "uploaded"
SAMPLES_FOLDER = PROJECT_ROOT / "dog_photos" / "samples"
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLES_FOLDER.mkdir(parents=True, exist_ok=True)

# Lazy/Cached Model Predictor
_predictor_instance: DogBreedPredictor | None = None


def get_predictor() -> DogBreedPredictor:
    """Returns the singleton predictor instance, initializing on first access."""
    global _predictor_instance
    if _predictor_instance is None:
        logger.info("Initializing DogBreedPredictor...")
        model_path = PROJECT_ROOT / "model" / "saved_models" / "best_model.keras"
        mapping_path = PROJECT_ROOT / "model" / "class_mapping" / "class_indices.json"
        _predictor_instance = DogBreedPredictor(
            model_path=model_path,
            class_mapping_path=mapping_path,
            enable_gradcam=True,
        )
    return _predictor_instance


def decode_base64_image(data_uri: str) -> bytes:
    """
    Decodes a base64 Data URI or raw base64 string to binary image bytes.

    Args:
        data_uri: Base64 data URI (e.g. 'data:image/jpeg;base64,...') or raw string.

    Returns:
        Decoded image bytes.

    Raises:
        ValueError: If input is empty, malformed, or cannot be base64-decoded.
    """
    if not data_uri or not isinstance(data_uri, str):
        raise ValueError("Image data must be a non-empty string.")
    if "," in data_uri:
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    try:
        return base64.b64decode(encoded)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Malformed base64 image encoding: {exc}") from exc


def get_sample_dogs() -> list[dict[str, str]]:
    """Discovers available sample dog images in the samples directory."""
    samples = []
    if SAMPLES_FOLDER.is_dir():
        for file in sorted(SAMPLES_FOLDER.glob("*.jpg")):
            # Filenames are typically '<Breed>_sample.jpg'
            raw_label = file.stem.replace("_sample", "")
            display_label = format_breed_display_name(raw_label)
            samples.append({
                "filename": file.name,
                "label": display_label,
            })
    return samples


# -------------------------------------------------------------------------
# Static Asset Serving Routes
# -------------------------------------------------------------------------
@app.route("/dog_photos/samples/<filename>")
def serve_sample(filename: str):
    """Serves verified sample dog images."""
    return send_from_directory(str(SAMPLES_FOLDER), filename)


@app.route("/dog_photos/uploaded/<filename>")
def serve_uploaded(filename: str):
    """Serves user-uploaded dog images."""
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    """Serves generated EDA, evaluation, and training charts."""
    return send_from_directory(str(OUTPUTS_FOLDER), filename)


# -------------------------------------------------------------------------
# User-Facing Web Pages
# -------------------------------------------------------------------------
@app.route("/")
def index():
    """Home page with image upload dropzone and sample selectors."""
    samples = get_sample_dogs()
    return render_template("index.html", samples=samples)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handles image upload or sample selection and renders prediction results."""
    if request.method == "GET":
        return redirect(url_for("index"))

    predictor = get_predictor()
    sample_name = request.form.get("sample_name")

    # Case 1: Prediction from existing sample image
    if sample_name:
        sample_path = SAMPLES_FOLDER / secure_filename(sample_name)
        if not sample_path.is_file():
            flash(f"Sample image '{sample_name}' was not found on the server.", "error")
            return redirect(url_for("index"))

        try:
            prediction_output = predictor.predict(sample_path, top_k=3, explain=True)
            uploaded_image_url = url_for("serve_sample", filename=sample_path.name)
            return render_template(
                "prediction.html",
                prediction=prediction_output,
                uploaded_image_url=uploaded_image_url,
            )
        except (ValueError, RuntimeError, TypeError, KeyError, OSError) as e:
            logger.error("Inference failed on sample %s: %s", sample_name, e)
            flash(f"Prediction failed: {e}", "error")
            return redirect(url_for("index"))

    # Case 2: Prediction from uploaded file
    if "image" not in request.files:
        flash("No image file provided in upload request.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not file or file.filename == "":
        flash("Please select an image file to upload.", "error")
        return redirect(url_for("index"))

    file_bytes = file.read()
    validation = ImageValidator.validate_bytes(file_bytes, filename=file.filename)
    if not validation.is_valid:
        flash(f"Upload validation failed: {validation.error_message}", "error")
        return redirect(url_for("index"))

    # Persist file with unique name
    ext = Path(file.filename).suffix.lower() or ".jpg"
    unique_filename = f"{uuid.uuid4().hex[:12]}{ext}"
    save_path = UPLOAD_FOLDER / unique_filename

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    try:
        prediction_output = predictor.predict(save_path, top_k=3, explain=True)
        uploaded_image_url = url_for("serve_uploaded", filename=unique_filename)
        return render_template(
            "prediction.html",
            prediction=prediction_output,
            uploaded_image_url=uploaded_image_url,
        )
    except (ValueError, RuntimeError, TypeError, KeyError, OSError) as e:
        logger.error("Inference failed on uploaded file %s: %s", unique_filename, e)
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    """Renders dataset statistics, train/val/test splits, and pipeline architecture."""
    return render_template("dashboard.html")


@app.route("/analytics")
def analytics():
    """Renders evaluation benchmarks, confusion matrices, and EDA plots."""
    return render_template("analytics.html")


@app.route("/model-info")
def model_info():
    """Renders deep neural network architecture specifications and layer hierarchy."""
    return render_template("model_info.html")


@app.route("/live")
def live():
    """Renders the real-time live camera and video stream dog breed classifier."""
    return render_template("live.html")


@app.route("/about")
def about():
    """Renders project documentation, engineering standards, and limitations."""
    return render_template("about.html")


# -------------------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Programmatic inference API endpoint.
    Accepts:
      1. multipart/form-data with 'image' file.
      2. application/json with 'image' (base64 Data URI or raw string), optional 'top_k', 'explain'.
    Returns JSON formatted prediction payload.
    """
    file_bytes: bytes | None = None
    filename: str | None = None
    top_k = 3
    explain = True

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image")
        if not image_b64:
            return jsonify({"status": "error", "message": "Missing 'image' key in JSON payload"}), 400
        try:
            file_bytes = decode_base64_image(image_b64)
            filename = "camera_capture.jpg"
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        top_k = int(payload.get("top_k", 3))
        explain = bool(payload.get("explain", True))
    elif "image" in request.files:
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"status": "error", "message": "Empty file uploaded"}), 400
        file_bytes = file.read()
        filename = file.filename
        top_k = int(request.form.get("top_k", 3))
        explain = request.form.get("explain", "true").lower() in ("true", "1", "yes")
    else:
        return jsonify({"status": "error", "message": "Missing 'image' in multipart upload or JSON body"}), 400

    validation = ImageValidator.validate_bytes(file_bytes, filename=filename)
    if not validation.is_valid:
        return jsonify({"status": "error", "message": validation.error_message}), 422

    predictor = get_predictor()
    try:
        prediction_output = predictor.predict(file_bytes, top_k=top_k, explain=explain)
        return jsonify({
            "status": "success",
            "data": prediction_output.to_dict(),
        }), 200
    except (ValueError, RuntimeError, TypeError, KeyError, OSError) as e:
        logger.error("API inference failed: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-live", methods=["POST"])
def api_predict_live():
    """
    High-frequency inference endpoint optimized for live video stream frames.
    Accepts application/json:
      {
        "image": "data:image/jpeg;base64,...",
        "top_k": 3
      }
    Runs rapid feedforward classification without Grad-CAM for low-latency (<100ms) streaming.
    """
    if not request.is_json:
        # Check if sent via multipart form
        image_b64 = request.form.get("image")
        if not image_b64 and "image" in request.files:
            file_bytes = request.files["image"].read()
        elif image_b64:
            try:
                file_bytes = decode_base64_image(image_b64)
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        else:
            return jsonify({"status": "error", "message": "Expected JSON payload with 'image' key"}), 400
    else:
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image")
        if not image_b64:
            return jsonify({"status": "error", "message": "Missing 'image' in JSON payload"}), 400
        try:
            file_bytes = decode_base64_image(image_b64)
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    if len(file_bytes) < 100:
        return jsonify({"status": "error", "message": "Frame payload too small"}), 422

    predictor = get_predictor()
    try:
        # Run inference with explain=False for maximum FPS and sub-100ms CPU latency
        prediction_output = predictor.predict(file_bytes, top_k=3, explain=False)
        out_dict = prediction_output.to_dict()
        return jsonify({
            "status": "success",
            "top_prediction": out_dict["top_prediction"],
            "top_k_predictions": out_dict["top_k_predictions"],
            "latency_ms": out_dict["latency_ms"],
        }), 200
    except (ValueError, RuntimeError, TypeError, KeyError, OSError) as e:
        logger.error("Live streaming inference error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint for container orchestration and uptime monitoring."""
    predictor_ready = _predictor_instance is not None
    return jsonify({
        "status": "healthy",
        "project": "Dog Breeds Classification",
        "project_id": "PRAICP-1007-120",
        "model_loaded": predictor_ready,
        "classes": 120,
    }), 200


# -------------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Uploaded file exceeds the maximum allowed size of 16 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("base.html", content="<div class='text-center py-20'><h1 class='text-4xl font-bold text-slate-800'>404 - Page Not Found</h1><p class='text-slate-500 mt-2'>The requested route does not exist.</p><a href='/' class='text-sky-600 font-semibold underline mt-4 inline-block'>Return to Home</a></div>"), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    logger.info("Starting Dog Breeds Classification web server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
