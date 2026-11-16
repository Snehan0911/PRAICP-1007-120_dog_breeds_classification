# =============================================================================
# CanineVision AI — Production Container Image
# Multi-purpose container supporting FastAPI backend & Streamlit portal
# =============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OS dependencies required by OpenCV and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY artifacts/class_mapping/ ./artifacts/class_mapping/
COPY pyproject.toml .

# Create non-root user for container security
RUN useradd -m -u 1000 appuser && \
    mkdir -p artifacts/models artifacts/metadata outputs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Default health check on FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
