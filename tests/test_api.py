"""Integration test suite for FastAPI network HTTP telemetry scoring endpoint."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.ml.inference import InferenceEngine


def test_score_telemetry_valid_payload(client: TestClient) -> None:
    """Assert valid JSON payload to POST /v1/telemetry/score returns 200 OK and anomaly scores."""
    valid_payload: dict[str, Any] = {
        "requests_per_minute": 120.0,
        "payload_bytes": 2500.0,
        "header_entropy": 3.5,
        "uri_depth": 3,
        "error_rate": 0.02,
    }

    response = client.post("/v1/telemetry/score", json=valid_payload)

    assert response.status_code == 200
    json_data = response.json()
    assert "anomaly_score" in json_data
    assert "is_anomalous" in json_data
    assert isinstance(json_data["anomaly_score"], float)
    assert isinstance(json_data["is_anomalous"], bool)


def test_score_telemetry_anomalous_payload(client: TestClient) -> None:
    """Assert extreme anomalous network telemetry payload flags traffic as anomalous."""
    anomalous_payload: dict[str, Any] = {
        "requests_per_minute": 50000.0,
        "payload_bytes": 100000.0,
        "header_entropy": 7.9,
        "uri_depth": 10,
        "error_rate": 0.99,
    }

    response = client.post("/v1/telemetry/score", json=anomalous_payload)

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["is_anomalous"] is True


def test_score_telemetry_invalid_payload_missing_fields(client: TestClient) -> None:
    """Assert payload with missing required fields returns 422 Unprocessable Entity."""
    invalid_payload: dict[str, Any] = {
        "requests_per_minute": 120.0,
        "payload_bytes": 2500.0,
    }

    response = client.post("/v1/telemetry/score", json=invalid_payload)

    assert response.status_code == 422


def test_score_telemetry_out_of_bounds_validation(client: TestClient) -> None:
    """Assert feature values outside Pydantic validation bounds return 422 Unprocessable Entity."""
    out_of_bounds_payload: dict[str, Any] = {
        "requests_per_minute": -10.0,
        "payload_bytes": 2500.0,
        "header_entropy": 9.5,  # Exceeds max 8.0
        "uri_depth": 3,
        "error_rate": 1.5,  # Exceeds max 1.0
    }

    response = client.post("/v1/telemetry/score", json=out_of_bounds_payload)

    assert response.status_code == 422


def test_health_check_endpoint(client: TestClient) -> None:
    """Assert GET /health returns 200 OK with status operational."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_inference_engine_file_not_found_raises(tmp_path: Path) -> None:
    """Assert FileNotFoundError is raised when scaler or ONNX model files are missing."""
    missing_scaler = tmp_path / "missing_scaler.pkl"
    missing_onnx = tmp_path / "missing_model.onnx"
    existing_file = tmp_path / "dummy.txt"
    existing_file.write_text("test")

    with pytest.raises(FileNotFoundError, match="Scaler file not found"):
        InferenceEngine(scaler_path=missing_scaler, onnx_path=existing_file)

    with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
        InferenceEngine(scaler_path=existing_file, onnx_path=missing_onnx)


def test_inference_engine_reset_instance() -> None:
    """Assert InferenceEngine singleton instance reset works correctly."""
    InferenceEngine.reset_instance()
    assert InferenceEngine._instance is None
