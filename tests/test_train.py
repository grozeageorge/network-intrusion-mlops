"""Unit tests for ML training pipeline, data preprocessing, and ONNX export utility functions."""

import csv
import tempfile
from collections.abc import Generator
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from src.ml.data import preprocess_data
from src.ml.train import (
    Autoencoder,
    export_onnx,
    export_scaler,
    main,
    train_model,
)


@pytest.fixture
def mock_csv_file() -> Generator[Path, None, None]:
    """Create a temporary CSV file with mock telemetry data, string columns, nan, and inf."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        tmp_path: Path = Path(tmp.name)
        writer = csv.writer(tmp)
        writer.writerow(
            ["Source IP", "Destination IP", "Timestamp", "Label", "f1", "f2", "f3", "f4", "f5"]
        )
        # Valid rows
        writer.writerow(
            ["192.168.1.1", "10.0.0.1", "2026-09-01 12:00:00", "BENIGN", 10.0, 100.0, 1.0, 2.0, 0.1]
        )
        writer.writerow(
            ["192.168.1.2", "10.0.0.2", "2026-09-01 12:01:00", "BENIGN", 20.0, 200.0, 2.0, 4.0, 0.2]
        )
        writer.writerow(
            ["192.168.1.3", "10.0.0.3", "2026-09-01 12:02:00", "ATTACK", 30.0, 300.0, 3.0, 6.0, 0.3]
        )
        # Invalid rows to be cleaned/dropped
        writer.writerow(
            [
                "192.168.1.4",
                "10.0.0.4",
                "2026-09-01 12:03:00",
                "BENIGN",
                float("nan"),
                150.0,
                2.5,
                3.0,
                0.15,
            ]
        )
        writer.writerow(
            [
                "192.168.1.5",
                "10.0.0.5",
                "2026-09-01 12:04:00",
                "BENIGN",
                25.0,
                float("inf"),
                2.0,
                3.0,
                0.1,
            ]
        )
        writer.writerow(
            [
                "192.168.1.6",
                "10.0.0.6",
                "2026-09-01 12:05:00",
                "BENIGN",
                15.0,
                120.0,
                float("-inf"),
                3.0,
                0.1,
            ]
        )

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()


def test_preprocess_data_drops_string_columns_and_handles_inf_nan(mock_csv_file: Path) -> None:
    """Assert preprocess_data drops string columns, drops inf/nan rows, and scales values in [0, 1]."""
    scaled_data, scaler = preprocess_data(mock_csv_file)

    assert isinstance(scaled_data, np.ndarray)
    assert isinstance(scaler, MinMaxScaler)
    assert scaled_data.dtype == np.float32

    # Verify 2 valid BENIGN rows remaining (ATTACK filtered out) and exactly 5 numerical feature columns
    assert scaled_data.shape == (2, 5)

    # Assert values are strictly scaled between 0.0 and 1.0
    assert np.all((scaled_data >= 0.0) & (scaled_data <= 1.0))
    assert np.isclose(scaled_data.min(), 0.0)
    assert np.isclose(scaled_data.max(), 1.0)


def test_preprocess_data_custom_drop_columns(tmp_path: Path) -> None:
    """Assert preprocess_data supports custom drop column lists."""
    csv_file: Path = tmp_path / "custom.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CustomMeta", "Unwanted", "val1", "val2"])
        writer.writerow(["meta_a", "skip", 1.0, 10.0])
        writer.writerow(["meta_b", "skip", 2.0, 20.0])

    scaled_data, scaler = preprocess_data(csv_file, drop_columns=["CustomMeta", "Unwanted"])
    assert isinstance(scaler, MinMaxScaler)
    assert scaled_data.shape == (2, 2)
    assert np.all((scaled_data >= 0.0) & (scaled_data <= 1.0))


def test_autoencoder_forward_pass() -> None:
    """Assert forward pass of Autoencoder preserves tensor shapes for dynamic feature sizes."""
    for num_features in (5, 8, 16):
        model: Autoencoder = Autoencoder(num_features=num_features)
        dummy_input: torch.Tensor = torch.randn(4, num_features, dtype=torch.float32)
        output: torch.Tensor = model(dummy_input)

        assert output.shape == (4, num_features)


def test_train_model_single_epoch(mock_csv_file: Path) -> None:
    """Assert train_model successfully trains Autoencoder instance over mini-batches."""
    scaled_data, _ = preprocess_data(mock_csv_file)

    model: Autoencoder = train_model(scaled_data, epochs=1, batch_size=16, seed=42)
    assert isinstance(model, Autoencoder)


def test_train_model_with_mlflow(mock_csv_file: Path, tmp_path: Path) -> None:
    """Assert train_model successfully logs metrics when wrapped in an active MLflow run."""
    scaled_data, _ = preprocess_data(mock_csv_file)

    db_path: Path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    with mlflow.start_run() as run:
        mlflow.log_params({"epochs": 2, "batch_size": 16, "learning_rate": 0.001})
        model: Autoencoder = train_model(
            scaled_data,
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
            seed=42,
        )
        assert isinstance(model, Autoencoder)

        client: mlflow.tracking.MlflowClient = mlflow.tracking.MlflowClient()
        metric_history = client.get_metric_history(run.info.run_id, "train_loss")
        assert len(metric_history) == 2


def test_export_scaler_and_onnx(tmp_path: Path) -> None:
    """Assert export_scaler and export_onnx write expected files to disk."""
    scaler: MinMaxScaler = MinMaxScaler()
    data: np.ndarray = np.random.rand(20, 5).astype(np.float32)
    scaler.fit(data)

    scaler_out: Path = tmp_path / "scaler.pkl"
    onnx_out: Path = tmp_path / "model.onnx"

    export_scaler(scaler, scaler_out)
    assert scaler_out.exists()

    model: Autoencoder = Autoencoder(num_features=5)
    export_onnx(model, onnx_out)
    assert onnx_out.exists()


def test_main_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert main() pipeline runs successfully, exports artifacts, and logs with MLflow."""
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = data_dir / "cicids2017_sample.csv"

    # Create small dataset for rapid test execution
    with open(sample_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Source IP", "Destination IP", "Timestamp", "Label", "f1", "f2", "f3", "f4", "f5"]
        )
        for i in range(10):
            writer.writerow(
                [
                    f"192.168.1.{i}",
                    "10.0.0.1",
                    "2026-09-01 12:00:00",
                    "BENIGN",
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    0.1,
                ]
            )

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    main()

    assert (models_dir / "scaler.pkl").exists()
    assert (models_dir / "autoencoder.onnx").exists()
