"""FastAPI application for the microservice."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request, status

from src.api.schemas import AnomalyScoreResponse, TelemetryPayload
from src.ml.inference import InferenceEngine

logger: logging.Logger = logging.getLogger("network_intrusion_mlops_api")


def get_inference_engine(request: Request) -> InferenceEngine:
    """Dependency provider for retrieving the active InferenceEngine instance.

    Args:
        request: FastAPI Request instance.

    Returns:
        Configured InferenceEngine singleton instance.
    """
    if not hasattr(request.app.state, "inference_engine"):
        request.app.state.inference_engine = InferenceEngine.get_instance()
    engine: InferenceEngine = request.app.state.inference_engine
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initializing singleton inference engine on startup."""
    logger.info("Initializing InferenceEngine during microservice startup...")
    engine: InferenceEngine = InferenceEngine.get_instance()
    app.state.inference_engine = engine
    yield
    logger.info("Shutting down microservice lifespan...")


app: FastAPI = FastAPI(
    title="Network Intrusion MLOps Microservice",
    description="Real-time HTTP telemetry anomaly scoring API powered by ONNX Runtime.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify microservice availability.

    Returns:
        JSON response with healthy status.
    """
    return {"status": "healthy"}


@app.post(
    "/v1/telemetry/score",
    response_model=AnomalyScoreResponse,
    status_code=status.HTTP_200_OK,
)
async def score_telemetry(
    payload: TelemetryPayload,
    engine: Annotated[InferenceEngine, Depends(get_inference_engine)],
) -> AnomalyScoreResponse:
    """Evaluate HTTP network telemetry data for network intrusion anomalies.

    Args:
        payload: Validated TelemetryPayload input object.
        engine: InferenceEngine dependency injected into route handler.

    Returns:
        AnomalyScoreResponse object containing reconstruction MSE score and anomaly flag.
    """
    return engine.predict(payload)
