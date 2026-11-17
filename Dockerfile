# =============================================================================
# Dog Breeds Classification – Production Dockerfile
# Project Name: Dog Breeds Classification
# Project ID: PRAICP-1007-120
# =============================================================================

FROM python:3.11-slim AS runtime

# System environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PORT=5000 \
    TF_ENABLE_ONEDNN_OPTS=1

WORKDIR /app

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Create a non-privileged system user for secure container execution
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/dog_photos/uploaded /app/log && \
    chown -R appuser:appuser /app

# Copy application codebase and artifacts
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser dog_photos/ ./dog_photos/
COPY --chown=appuser:appuser outputs/ ./outputs/
COPY --chown=appuser:appuser docs/ ./docs/

# Switch to non-root user
USER appuser

# Expose web service ports (7860 for Hugging Face Spaces, 5000 for standard Docker)
EXPOSE 7860 5000

# Container healthcheck targeting /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || curl -f http://localhost:5000/health || exit 1

# Production WSGI server command using dynamic port binding
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app"]
