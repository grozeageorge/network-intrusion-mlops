"""Data ingestion and preprocessing pipelines for network telemetry datasets."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger: logging.Logger = logging.getLogger("network_intrusion_mlops_data")


def preprocess_data(
    csv_path: str | Path,
    drop_columns: list[str] | None = None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Load, clean, and scale tabular telemetry data from a CSV file.

    Args:
        csv_path: Path to the raw CSV dataset.
        drop_columns: Optional list of metadata columns to discard. If None,
            defaults to standard CICIDS2017 metadata and label columns.

    Returns:
        tuple[np.ndarray, MinMaxScaler]: Cleaned float32 scaled array of shape (samples, features)
            and the fitted MinMaxScaler instance for serialization.
    """
    if drop_columns is None:
        drop_columns = [
            "Flow ID",
            "Source IP",
            "Src IP",
            "Source Port",
            "Src Port",
            "Destination IP",
            "Dst IP",
            "Destination Port",
            "Dst Port",
            "Protocol",
            "Timestamp",
            "Label",
        ]

    df: pd.DataFrame = pd.read_csv(csv_path)

    if "Label" in df.columns:
        benign_mask: pd.Series = df["Label"].astype(str).str.strip().str.upper() == "BENIGN"
        df = cast(pd.DataFrame, df.loc[benign_mask])

    df = cast(pd.DataFrame, df.drop(columns=drop_columns, errors="ignore"))
    df = cast(pd.DataFrame, df.select_dtypes(include=[np.number]))
    df = cast(pd.DataFrame, df.replace([np.inf, -np.inf], np.nan).dropna())

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
