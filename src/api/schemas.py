"""Pydantic data models for network telemetry payload validation and scoring responses."""

from pydantic import BaseModel, Field


class TelemetryPayload(BaseModel):
    """HTTP network traffic telemetry payload for anomaly scoring."""

    features: list[float] = Field(
        ...,
        description="Vector of numerical telemetry feature values for anomaly scoring",
        min_length=1,
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
