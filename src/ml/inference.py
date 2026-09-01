"""ONNX runtime inference engine for network traffic anomaly detection."""

import logging
from pathlib import Path
from typing import ClassVar, Optional, cast

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.api.schemas import AnomalyScoreResponse, TelemetryPayload

logger: logging.Logger = logging.getLogger("network_intrusion_mlops_inference")


class InferenceEngine:
    """Singleton ONNX Runtime Inference Engine for network HTTP traffic scoring."""

    _instance: ClassVar[Optional["InferenceEngine"]] = None
    ANOMALY_THRESHOLD: ClassVar[float] = 0.05

    def __init__(self, scaler_path: Path, onnx_path: Path) -> None:
        """Initialize scaler and ONNX InferenceSession.

        Args:
            scaler_path: Path to serialized StandardScaler pkl file.
            onnx_path: Path to serialized ONNX model file.

        Raises:
            FileNotFoundError: If scaler or ONNX model files do not exist.
        """
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")

        self.scaler: StandardScaler | MinMaxScaler = joblib.load(scaler_path)
        self.session: ort.InferenceSession = ort.InferenceSession(str(onnx_path))
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name
        logger.info("InferenceEngine initialized with model %s", onnx_path)

    @classmethod
    def get_instance(
        cls,
        scaler_path: Path = Path("models/scaler.pkl"),
        onnx_path: Path = Path("models/autoencoder.onnx"),
    ) -> "InferenceEngine":
        """Retrieve or create the singleton InferenceEngine instance.

        Args:
            scaler_path: Path to StandardScaler pkl file.
            onnx_path: Path to ONNX model file.

        Returns:
            Singleton InferenceEngine instance.
        """
        if cls._instance is None:
            cls._instance = cls(scaler_path=scaler_path, onnx_path=onnx_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for unit testing)."""
        cls._instance = None

    def predict(self, payload: TelemetryPayload) -> AnomalyScoreResponse:
        """Process telemetry payload, compute ONNX reconstruction MSE loss, and classify.

        Args:
            payload: Validated TelemetryPayload object.

        Returns:
            AnomalyScoreResponse containing MSE score and anomaly boolean flag.
        """
        raw_features: np.ndarray = np.array(
            [
                [
                    payload.requests_per_minute,
                    payload.payload_bytes,
                    payload.header_entropy,
                    float(payload.uri_depth),
                    payload.error_rate,
                ]
            ],
            dtype=np.float32,
        )

        scaled_features: np.ndarray = self.scaler.transform(raw_features).astype(np.float32)

        onnx_outputs: list[np.ndarray] = cast(
            list[np.ndarray],
            self.session.run(
                [self.output_name],
                {self.input_name: scaled_features},
            ),
        )
        reconstructed: np.ndarray = onnx_outputs[0]

        mse_loss: float = float(np.mean((scaled_features - reconstructed) ** 2))
        is_anomalous: bool = mse_loss > self.ANOMALY_THRESHOLD

        return AnomalyScoreResponse(
            anomaly_score=round(mse_loss, 6),
            is_anomalous=is_anomalous,
        )
