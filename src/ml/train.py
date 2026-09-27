"""Module for training PyTorch Autoencoder model for network HTTP anomaly detection.

Preprocesses tabular telemetry data, trains an autoencoder neural network,
tracks experiments using MLflow, exports the trained model to ONNX format,
and serializes the fitted feature scaler.
"""

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger("network_intrusion_mlops_ml")


class Autoencoder(nn.Module):
    """PyTorch Autoencoder for network HTTP anomaly detection.

    Encoder topology: num_features -> max(1, num_features // 2) -> max(1, num_features // 4)
    Decoder topology: max(1, num_features // 4) -> max(1, num_features // 2) -> num_features
    """

    def __init__(self, num_features: int = 5) -> None:
        """Initialize encoder and decoder network layers.

        Args:
            num_features: Number of input features in tabular dataset.
        """
        super().__init__()
        self.num_features: int = num_features
        hidden_dim1: int = max(1, num_features // 2)
        hidden_dim2: int = max(1, num_features // 4)

        self.encoder: nn.Sequential = nn.Sequential(
            nn.Linear(num_features, hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
        )
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Reconstructed output tensor of shape (batch_size, num_features).
        """
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded


def preprocess_data(
    csv_path: str | Path,
    drop_columns: list[str] | None = None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Load, clean, and scale tabular telemetry data from a CSV file.

    Args:
        csv_path: Path to the raw CSV dataset.
        drop_columns: Optional list of metadata columns to discard.

    Returns:
        tuple[np.ndarray, MinMaxScaler]: Cleaned float32 scaled array of shape (samples, features)
            and the fitted MinMaxScaler instance for serialization.
    """
    if drop_columns is None:
        drop_columns = ["Source IP", "Destination IP", "Timestamp", "Label"]

    df: pd.DataFrame = pd.read_csv(csv_path)

    df = df.drop(columns=drop_columns, errors="ignore")
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    scaler: MinMaxScaler = MinMaxScaler()
    scaled_data: np.ndarray = scaler.fit_transform(df.to_numpy()).astype(np.float32)

    logger.info(
        "Preprocessed dataset %s into shape %s with values in [%.4f, %.4f]",
        csv_path,
        scaled_data.shape,
        scaled_data.min(),
        scaled_data.max(),
    )
    return scaled_data, scaler


def train_model(
    data: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    model: Autoencoder | None = None,
) -> Autoencoder:
    """Train PyTorch Autoencoder on scaled normal traffic data and log metrics to MLflow.

    Args:
        data: Preprocessed telemetry dataset.
        epochs: Training epoch count.
        batch_size: DataLoader mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed for PyTorch weight initialization.
        model: Optional pre-initialized Autoencoder instance. If None, initializes
            Autoencoder(num_features=data.shape[1]).

    Returns:
        Trained Autoencoder model instance.
    """
    torch.manual_seed(seed)

    tensor_data: torch.Tensor = torch.from_numpy(data)
    dataset: TensorDataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = Autoencoder(num_features=int(data.shape[1]))

    criterion: nn.MSELoss = nn.MSELoss()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting Autoencoder model training for %d epochs...", epochs)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs: torch.Tensor = model(batch_x)
            loss: torch.Tensor = criterion(outputs, batch_y)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss: float = total_loss / len(dataset)
        if mlflow.active_run() is not None:
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info("Epoch [%d/%d] - Loss (MSE): %.6f", epoch, epochs, avg_loss)

    logger.info("Model training complete.")
    return model


def export_scaler(scaler: StandardScaler | MinMaxScaler, output_path: Path) -> None:
    """Save fitted scaler to disk using joblib.

    Args:
        scaler: Fitted StandardScaler or MinMaxScaler instance.
        output_path: Destination path for serialized file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    logger.info("Scaler successfully saved to %s", output_path)


def export_onnx(model: Autoencoder, output_path: Path) -> None:
    """Export PyTorch Autoencoder model to ONNX format with dynamic batch sizing.

    Args:
        model: Trained Autoencoder module.
        output_path: Destination path for ONNX file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input: torch.Tensor = torch.randn(1, model.num_features, dtype=torch.float32)
    dynamic_axes: dict[str, dict[int, str]] = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    logger.info("ONNX model successfully exported to %s", output_path)


def main() -> None:
    """Execute machine learning training pipeline, MLflow tracking, and ONNX export."""
    models_dir: Path = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path: Path = models_dir / "scaler.pkl"
    onnx_path: Path = models_dir / "autoencoder.onnx"
    data_path: Path = Path("data/raw/cicids2017_sample.csv")

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3

    logger.info("Starting ML pipeline with dataset: %s", data_path)
    scaled_data, scaler = preprocess_data(data_path)
    num_features: int = int(scaled_data.shape[1])
    autoencoder: Autoencoder = Autoencoder(num_features=num_features)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )

        model: Autoencoder = train_model(
            scaled_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model=autoencoder,
        )

        export_scaler(scaler, scaler_path)
        export_onnx(model, onnx_path)

        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(onnx_path))

    logger.info("ML training, artifact export, and MLflow tracking completed successfully.")


if __name__ == "__main__":
    main()
