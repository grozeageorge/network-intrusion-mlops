"""Module for training PyTorch Autoencoder model for network HTTP anomaly detection.

Generates synthetic telemetry data, trains an autoencoder neural network,
exports the trained model to ONNX format, and saves the feature scaler.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
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

    Encoder topology: 5 -> 3 -> 2
    Decoder topology: 2 -> 3 -> 5
    """

    def __init__(self) -> None:
        """Initialize encoder and decoder network layers."""
        super().__init__()
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Linear(5, 3),
            nn.LeakyReLU(0.1),
            nn.Linear(3, 2),
        )
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(2, 3),
            nn.LeakyReLU(0.1),
            nn.Linear(3, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, 5).

        Returns:
            Reconstructed output tensor of shape (batch_size, 5).
        """
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded


def generate_synthetic_telemetry(num_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate synthetic normal network HTTP telemetry data.

    Features (5 dimensions):
    1. requests_per_minute: float ~ Normal(120, 20)
    2. payload_bytes: float ~ Normal(2500, 500)
    3. header_entropy: float ~ Normal(3.5, 0.4)
    4. uri_depth: float ~ Normal(3.0, 1.0)
    5. error_rate: float ~ Beta(0.5, 20.0)

    Args:
        num_samples: Number of telemetry samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Numpy array of shape (num_samples, 5) with float32 data.
    """
    np.random.seed(seed)

    requests_per_minute = np.clip(
        np.random.normal(loc=120.0, scale=20.0, size=num_samples),
        a_min=1.0,
        a_max=None,
    )
    payload_bytes = np.clip(
        np.random.normal(loc=2500.0, scale=500.0, size=num_samples),
        a_min=10.0,
        a_max=None,
    )
    header_entropy = np.clip(
        np.random.normal(loc=3.5, scale=0.4, size=num_samples),
        a_min=0.0,
        a_max=8.0,
    )
    uri_depth = np.clip(
        np.random.normal(loc=3.0, scale=1.0, size=num_samples),
        a_min=1.0,
        a_max=10.0,
    )
    error_rate = np.clip(
        np.random.beta(a=0.5, b=20.0, size=num_samples),
        a_min=0.0,
        a_max=1.0,
    )

    data: np.ndarray = np.column_stack(
        [
            requests_per_minute,
            payload_bytes,
            header_entropy,
            uri_depth,
            error_rate,
        ]
    ).astype(np.float32)

    logger.info(
        "Generated %d synthetic normal telemetry samples with shape %s", num_samples, data.shape
    )
    return data


def train_model(
    data: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Autoencoder:
    """Train PyTorch Autoencoder on scaled normal traffic data.

    Args:
        data: Preprocessed telemetry dataset.
        epochs: Training epoch count.
        batch_size: DataLoader mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed for PyTorch weight initialization.

    Returns:
        Trained Autoencoder model instance.
    """
    torch.manual_seed(seed)

    tensor_data: torch.Tensor = torch.from_numpy(data)
    dataset: TensorDataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model: Autoencoder = Autoencoder()
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
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info("Epoch [%d/%d] - Loss (MSE): %.6f", epoch, epochs, avg_loss)

    logger.info("Model training complete.")
    return model


def export_scaler(scaler: StandardScaler, output_path: Path) -> None:
    """Save fitted StandardScaler to disk using joblib.

    Args:
        scaler: Fitted StandardScaler instance.
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

    dummy_input: torch.Tensor = torch.randn(1, 5, dtype=torch.float32)
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
    """Execute machine learning training pipeline and ONNX export."""
    models_dir: Path = Path("models")
    scaler_path: Path = models_dir / "scaler.pkl"
    onnx_path: Path = models_dir / "autoencoder.onnx"

    raw_data: np.ndarray = generate_synthetic_telemetry(num_samples=5000, seed=42)

    scaler: StandardScaler = StandardScaler()
    scaled_data: np.ndarray = scaler.fit_transform(raw_data).astype(np.float32)

    export_scaler(scaler, scaler_path)

    model: Autoencoder = train_model(scaled_data, epochs=50, batch_size=64, learning_rate=1e-3)

    export_onnx(model, onnx_path)

    logger.info("ML training and ONNX export completed successfully.")


if __name__ == "__main__":
    main()
