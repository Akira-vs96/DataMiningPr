"""Shared data validation and reporting for practicals 10–15, variant 1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from breast_cancer_labs.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def markdown_table(table: pd.DataFrame) -> str:
    def cell(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value).replace("|", "/").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in table.itertuples(index=False, name=None)],
    ])


def regression_metrics(actual, predicted) -> dict:
    return {"mae": mean_absolute_error(actual, predicted),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": r2_score(actual, predicted)}


def save_environment(output: Path, sources: list[Path], **settings) -> None:
    packages = ["numpy", "pandas", "matplotlib", "scikit-learn", "statsmodels", "shap", "lime", "river", "networkx"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    payload = {"python": platform.python_version(), "packages": versions,
               "variant": 1, "random_state": 42,
               "sources_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               **settings}
    (output / "environment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_daily_electricity() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DIR / "elecdaily.csv")
    if len(raw) != 365 or raw[["Demand", "Temperature", "WorkDay"]].isna().any().any():
        raise ValueError("Expected 365 complete daily electricity observations")
    data = raw[["Demand", "Temperature", "WorkDay"]].copy()
    data.index = pd.date_range("2014-01-01", periods=len(data), freq="D", name="date")
    return data


def electricity_features(data: pd.DataFrame) -> pd.DataFrame:
    """Weather must be observed or forecast separately for future-day prediction."""
    day = data.index.dayofyear.to_numpy()
    return pd.DataFrame({
        "temperature_c": data.Temperature,
        "temperature_squared": data.Temperature ** 2,
        "workday": data.WorkDay,
        "annual_sin": np.sin(2 * np.pi * day / 365.25),
        "annual_cos": np.cos(2 * np.pi * day / 365.25),
        "weekly_sin": np.sin(2 * np.pi * data.index.dayofweek / 7),
        "weekly_cos": np.cos(2 * np.pi * data.index.dayofweek / 7),
    }, index=data.index)
