from fastapi import FastAPI

from app.config import get_settings
from app.domain.schemas import HealthResponse

settings = get_settings()
app = FastAPI(title="Hybrid RAG", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
    )


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    return HealthResponse(
        status="ready",
        service=settings.app_name,
        environment=settings.environment,
    )
