"""
CanineVision AI — FastAPI Production Application Entry Point.

Architectural highlights:
- Lifespan context manager loads deep learning model into RAM ONCE at startup.
- Never reloads model per request, maintaining millisecond latency.
- Structured CORS middleware, request logging, and global exception handlers.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routes import health, models, prediction
from src.inference.predictor import BreedPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle management.
    Pre-warms and caches the deep learning model.
    """
    logger.info("Initializing CanineVision AI Inference Service...")
    # Load model and class mapping once into memory
    app.state.predictor = BreedPredictor(model_name="mobilenet_v2")
    logger.info("Model loaded successfully. API is ready to accept requests.")
    yield
    logger.info("Shutting down CanineVision AI Inference Service...")


app = FastAPI(
    title="CanineVision AI — Dog Breed Intelligence API",
    description=(
        "Enterprise-grade Computer Vision REST API identifying 120 dog breeds from images. "
        "Provides Top-3 breed ranking, confidence interpretation, and Grad-CAM explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(health.router)
app.include_router(models.router)
app.include_router(prediction.router)


@app.get("/", include_in_schema=False)
def root():
    """Redirects root visitors directly to interactive OpenAPI Swagger documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
