"""
CanineVision AI — Inference Input Image Validator.

Guards the prediction engine against:
- Malformed / corrupted image payloads
- Non-standard image dimensions (e.g. 0-pixel width)
- Extremely large payloads exceeding memory safety limits
- Unsupported file extensions
"""

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION = 32


class ImageValidator:
    """
    Validates byte streams and files prior to deep learning preprocessing.
    """

    @staticmethod
    def validate_bytes(image_bytes: bytes) -> tuple[bool, str | None, Image.Image | None]:
        """
        Validates raw image bytes from HTTP upload or memory.

        Returns:
            Tuple of (is_valid, error_message, PIL Image if valid).
        """
        if not image_bytes:
            return False, "Uploaded image file is empty (0 bytes).", None

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit (10 MB).", None

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_format = (img.format or "").upper()

            if img_format not in SUPPORTED_FORMATS:
                return False, f"Unsupported format '{img_format}'. Supported formats: JPEG, PNG, WEBP, BMP.", None

            w, h = img.size
            if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                return False, f"Image resolution ({w}x{h}) is too small (minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}).", None

            # Verify image integrity
            img.verify()

            # Re-open after verify() closes it
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return True, None, img

        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            return False, f"Corrupted or invalid image file: {str(e)}", None

    @staticmethod
    def validate_file(filepath: str | Path) -> tuple[bool, str | None]:
        """Validates an image file on the local filesystem."""
        p = Path(filepath)
        if not p.exists():
            return False, f"File does not exist: {p}"
        try:
            with open(p, "rb") as f:
                valid, msg, _ = ImageValidator.validate_bytes(f.read())
                return valid, msg
        except Exception as e:
            return False, str(e)
