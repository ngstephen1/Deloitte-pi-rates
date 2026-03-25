"""
Utility functions: logging, data handling, normalization helpers.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from . import config

# ============================================================================
# LOGGING
# ============================================================================
def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with standard format."""
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(config.LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


LOGGER = setup_logger(__name__)


# ============================================================================
# DATA NORMALIZATION & SCALING
# ============================================================================
def normalize_to_01(values: np.ndarray) -> np.ndarray:
    """
    Normalize values to [0, 1] range using MinMaxScaler.
    Handles NaN/inf gracefully.
    """
    values = np.asarray(values).flatten()
    # Replace inf with NaN
    values = np.where(np.isinf(values), np.nan, values)

    if np.all(np.isnan(values)):
        return np.zeros_like(values)

    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.zeros_like(values)

    scaler = MinMaxScaler(feature_range=(0, 1))
    result = np.zeros_like(values, dtype=float)
    
    # Fit and transform only valid values
    scaled_values = scaler.fit_transform(values[valid_mask].reshape(-1, 1))
    # Ensure result is 1D
    scaled_values = scaled_values.ravel()
    result[valid_mask] = scaled_values
    result[~valid_mask] = np.nan

    return result


def standardize_features(values: np.ndarray) -> np.ndarray:
    """
    Standardize values to mean=0, std=1 using StandardScaler.
    Handles NaN gracefully.
    """
    values = np.asarray(values).flatten()

    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.zeros_like(values)

    scaler = StandardScaler()
    result = np.zeros_like(values, dtype=float)
    result[valid_mask] = scaler.fit_transform(values[valid_mask].reshape(-1, 1)).ravel()
    result[~valid_mask] = np.nan

    return result


# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values.
    Returns DataFrame with column, count, and percentage missing.
    """
    missing = df.isnull().sum()
    missing_pct = 100 * df.isnull().sum() / len(df)
    summary = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    })
    return summary[summary["missing_count"] > 0].reset_index(drop=True)


def detect_basic_outliers(series: pd.Series, method: str = "iqr") -> Tuple[np.ndarray, dict]:
    """
    Detect outliers in a numeric series using IQR or Z-score.

    Args:
        series: Numeric series
        method: 'iqr' or 'zscore'

    Returns:
        Tuple of (outlier_mask, stats_dict)
    """
    series = series.dropna()

    if len(series) == 0:
        return np.array([], dtype=bool), {}

    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        stats = {
            "method": "iqr",
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_mask.sum(),
        }
    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        outlier_mask = z_scores > 3
        stats = {
            "method": "zscore",
            "mean": mean,
            "std": std,
            "threshold": 3,
            "outlier_count": outlier_mask.sum(),
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return outlier_mask.values, stats


# ============================================================================
# FILE I/O
# ============================================================================
def save_csv(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """Save DataFrame to CSV with consistent formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    LOGGER.info(f"Saved {len(df)} rows to {filepath}")


def load_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load DataFrame from CSV with error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, **kwargs)
    LOGGER.info(f"Loaded {len(df)} rows from {filepath}")
    return df


# ============================================================================
# FEATURE VALIDATION
# ============================================================================
def validate_required_columns(df: pd.DataFrame, required_cols: list) -> None:
    """Check that all required columns are present."""
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Return list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Return list of categorical column names."""
    return df.select_dtypes(include=["object"]).columns.tolist()
