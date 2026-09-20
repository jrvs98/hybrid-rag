from typing import Optional

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.domain.schemas import HealthResponse, QueryRequest, QueryResponse
from app.pipeline import RAGPipeline

settings = get_settings()
app = FastAPI(title="Hybrid RAG", version="0.1.0")
app.state.query_pipeline: Optional[RAGPipeline] = None


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


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = app.state.query_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="query pipeline is not configured")
    return pipeline.query(request)
