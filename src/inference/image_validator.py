"""
Image Upload Validator for Dog Breeds Classification.
Project: Dog Breeds Classification (PRAICP-1007-120)

Validates user-submitted images against corruption, file size boundaries,
unsupported formats, and dimension thresholds before model inference.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PIL import Image, UnidentifiedImageError


@dataclass
class ValidationResult:
    """Represents the outcome of image file validation."""
    is_valid: bool
    error_message: str | None = None
    width: int = 0
    height: int = 0
    format: str = ""
    size_bytes: int = 0


class ImageValidator:
    """
    Validates uploaded images for format, size, dimensions, and byte integrity.
    """

    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ALLOWED_FORMATS: ClassVar[set[str]] = {"JPEG", "PNG", "WEBP", "BMP"}
    MAX_FILE_SIZE_BYTES: ClassVar[int] = 16 * 1024 * 1024  # 16 MB
    MIN_FILE_SIZE_BYTES: ClassVar[int] = 100  # 100 bytes
    MIN_DIMENSION: ClassVar[int] = 32
    MAX_DIMENSION: ClassVar[int] = 8192

    @classmethod
    def validate_file(cls, file_path: str | Path) -> ValidationResult:
        """
        Validates an image file located on disk.

        Args:
            file_path: Path to the target image file.

        Returns:
            ValidationResult with status and metadata or error explanation.
        """
        path = Path(file_path)
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                error_message=f"File not found: {path.name}",
            )

        # Check extension
        ext = path.suffix.lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(cls.ALLOWED_EXTENSIONS))}",
            )

        # Check file size
        file_size = path.stat().st_size
        if file_size < cls.MIN_FILE_SIZE_BYTES:
            return ValidationResult(
                is_valid=False,
                error_message=f"File is empty or too small ({file_size} bytes). Minimum required is {cls.MIN_FILE_SIZE_BYTES} bytes.",
                size_bytes=file_size,
            )
        if file_size > cls.MAX_FILE_SIZE_BYTES:
            max_mb = cls.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            return ValidationResult(
                is_valid=False,
                error_message=f"File size exceeds maximum limit of {max_mb} MB ({file_size / (1024 * 1024):.2f} MB).",
                size_bytes=file_size,
            )

        # Validate byte integrity and dimensions
        try:
            with Image.open(path) as img:
                img.verify()

            # Re-open to read dimensions and format (verify() closes/invalidates the image)
            with Image.open(path) as img:
                width, height = img.size
                img_format = img.format or ""

                if img_format.upper() not in cls.ALLOWED_FORMATS:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Unsupported image encoding format '{img_format}'.",
                        size_bytes=file_size,
                        format=img_format,
                    )

                if width < cls.MIN_DIMENSION or height < cls.MIN_DIMENSION:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Image dimensions ({width}x{height}) are below minimum {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION} pixels.",
                        width=width,
                        height=height,
                        format=img_format,
                        size_bytes=file_size,
                    )

                if width > cls.MAX_DIMENSION or height > cls.MAX_DIMENSION:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Image dimensions ({width}x{height}) exceed maximum allowed {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION} pixels.",
                        width=width,
                        height=height,
                        format=img_format,
                        size_bytes=file_size,
                    )

                return ValidationResult(
                    is_valid=True,
                    width=width,
                    height=height,
                    format=img_format,
                    size_bytes=file_size,
                )

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image file is corrupted or could not be decoded: {exc}",
                size_bytes=file_size,
            )

    @classmethod
    def validate_bytes(cls, data: bytes, filename: str | None = None) -> ValidationResult:
        """
        Validates in-memory byte buffer from an HTTP multipart upload.

        Args:
            data: Raw binary contents of the image.
            filename: Optional original upload filename to check file extension.

        Returns:
            ValidationResult with status and metadata or error explanation.
        """
        if not data:
            return ValidationResult(is_valid=False, error_message="Empty upload payload.")

        size_bytes = len(data)
        if size_bytes < cls.MIN_FILE_SIZE_BYTES:
            return ValidationResult(
                is_valid=False,
                error_message=f"Upload too small ({size_bytes} bytes).",
                size_bytes=size_bytes,
            )
        if size_bytes > cls.MAX_FILE_SIZE_BYTES:
            max_mb = cls.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            return ValidationResult(
                is_valid=False,
                error_message=f"Upload exceeds maximum allowable limit of {max_mb} MB.",
                size_bytes=size_bytes,
            )

        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in cls.ALLOWED_EXTENSIONS:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported file extension '{ext}'.",
                    size_bytes=size_bytes,
                )

        try:
            stream = io.BytesIO(data)
            with Image.open(stream) as img:
                img.verify()

            stream.seek(0)
            with Image.open(stream) as img:
                width, height = img.size
                img_format = img.format or ""

                if img_format.upper() not in cls.ALLOWED_FORMATS:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Unsupported image encoding format '{img_format}'.",
                        size_bytes=size_bytes,
                        format=img_format,
                    )

                if width < cls.MIN_DIMENSION or height < cls.MIN_DIMENSION:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Image dimensions ({width}x{height}) are below minimum {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION} pixels.",
                        width=width,
                        height=height,
                        format=img_format,
                        size_bytes=size_bytes,
                    )

                return ValidationResult(
                    is_valid=True,
                    width=width,
                    height=height,
                    format=img_format,
                    size_bytes=size_bytes,
                )

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return ValidationResult(
                is_valid=False,
                error_message=f"Corrupted or invalid image data: {exc}",
                size_bytes=size_bytes,
            )
