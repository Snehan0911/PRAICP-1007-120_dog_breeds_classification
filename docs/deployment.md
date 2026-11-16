# CanineVision AI — Deployment & Infrastructure Guide

## 1. Local Containerized Run (Docker Compose)
The recommended local production setup uses `docker-compose`:

```bash
docker-compose up --build -d
```

- **Backend:** `http://localhost:8000`
- **Frontend:** `http://localhost:8501`

To view container logs:
```bash
docker-compose logs -f
```

To stop containers:
```bash
docker-compose down
```

## 2. Standalone Container Execution
```bash
# Build the image
docker build -t caninevision-ai:latest .

# Run FastAPI backend
docker run -d -p 8000:8000 --name caninevision-api caninevision-ai:latest

# Run Streamlit frontend
docker run -d -p 8501:8501 --name caninevision-ui caninevision-ai:latest streamlit run app/app.py --server.port=8501
```

## 3. Kubernetes / Cloud Run Readiness
- **Stateless Architecture:** The application stores no persistent user sessions in memory.
- **Healthcheck:** Standard HTTP `/health` probe compatible with Kubernetes liveness and readiness probes.
- **Non-root Security:** Container runs under dedicated UID 1000 `appuser`.
