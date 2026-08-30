"""Pydantic data models for network telemetry payload validation and scoring responses."""

from pydantic import BaseModel, Field


class TelemetryPayload(BaseModel):
    """HTTP network traffic telemetry payload for anomaly scoring."""

    requests_per_minute: float = Field(
        ...,
        description="Number of HTTP requests per minute",
        ge=0.0,
    )
    payload_bytes: float = Field(
        ...,
        description="Average HTTP payload size in bytes",
        ge=0.0,
    )
    header_entropy: float = Field(
        ...,
        description="Shannon entropy of HTTP headers",
        ge=0.0,
        le=8.0,
    )
    uri_depth: int = Field(
        ...,
        description="Depth of HTTP URI path segments",
        ge=0,
    )
    error_rate: float = Field(
        ...,
        description="HTTP error status rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class AnomalyScoreResponse(BaseModel):
    """Response schema containing anomaly score and classification flag."""

    anomaly_score: float = Field(
        ...,
        description="MSE reconstruction error calculated by the Autoencoder model",
    )
    is_anomalous: bool = Field(
        ...,
        description="Flag indicating whether traffic is classified as an anomaly (MSE > 0.05)",
    )
