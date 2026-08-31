"""Unit tests for ML training pipeline and ONNX export utility functions."""

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.ml.train import (
    Autoencoder,
    export_onnx,
    export_scaler,
    generate_synthetic_telemetry,
    train_model,
)


def test_autoencoder_forward_pass() -> None:
    """Assert forward pass of Autoencoder preserves tensor shapes."""
    model = Autoencoder()
    dummy_input = torch.randn(4, 5, dtype=torch.float32)
    output = model(dummy_input)

    assert output.shape == (4, 5)


def test_generate_synthetic_telemetry() -> None:
    """Assert synthetic telemetry data generator returns expected array shape and dtype."""
    data = generate_synthetic_telemetry(num_samples=100, seed=123)

    assert isinstance(data, np.ndarray)
    assert data.shape == (100, 5)
    assert data.dtype == np.float32


def test_train_model_single_epoch() -> None:
    """Assert train_model successfully trains Autoencoder instance over mini-batches."""
    data = generate_synthetic_telemetry(num_samples=50, seed=42)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data).astype(np.float32)

    model = train_model(scaled_data, epochs=1, batch_size=16, seed=42)
    assert isinstance(model, Autoencoder)


def test_export_scaler_and_onnx(tmp_path: Path) -> None:
    """Assert export_scaler and export_onnx write expected files to disk."""
    scaler = StandardScaler()
    data = np.random.randn(20, 5).astype(np.float32)
    scaler.fit(data)

    scaler_out = tmp_path / "scaler.pkl"
    onnx_out = tmp_path / "model.onnx"

    export_scaler(scaler, scaler_out)
    assert scaler_out.exists()

    model = Autoencoder()
    export_onnx(model, onnx_out)
    assert onnx_out.exists()
