"""Pytest configuration and shared test fixtures for the Microservice."""

from collections.abc import Generator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from src.api.main import app
from src.ml.train import Autoencoder, export_onnx


@pytest.fixture(scope="session", autouse=True)
def ensure_models_exist() -> None:
    """Ensure trained model artifacts exist before executing integration tests."""
    scaler_path: Path = Path("models/scaler.pkl")
    onnx_path: Path = Path("models/autoencoder.onnx")
    if not (scaler_path.exists() and onnx_path.exists()):
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        scaler: MinMaxScaler = MinMaxScaler()
        scaler.fit([[0.0] * 5, [1.0] * 5])
        joblib.dump(scaler, scaler_path)

        model: Autoencoder = Autoencoder(num_features=5)
        export_onnx(model, onnx_path)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient instance bound to the application.

    Yields:
        Configured FastAPI TestClient instance.
    """
    with TestClient(app) as test_client:
        yield test_client
