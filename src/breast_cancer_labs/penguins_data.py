"""Load Penguins and remove incomplete rows without imputing any values."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import PENGUINS_ARCHIVE


DATA_MEMBER = "Penguin Species Prediction Dataset.csv"
NUMERIC_FEATURES = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURE_NAMES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]


def load_raw_penguins_dataset(archive_path: Path = PENGUINS_ARCHIVE) -> pd.DataFrame:
    """Validate the source table, preserving missing values for the removal audit."""
    with ZipFile(archive_path) as archive:
        with archive.open(DATA_MEMBER) as source:
            data = pd.read_csv(source)

    expected = ["species", *FEATURE_NAMES]
    if set(data.columns) != set(expected):
        raise ValueError(f"Expected columns {expected}, received {data.columns.tolist()}")
    data = data[expected].copy()
    if data.empty:
        raise ValueError("Penguins source table must not be empty")
    if set(data["species"].dropna().unique()) != set(SPECIES):
        raise ValueError(f"Expected species {SPECIES}")
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(f"Infinite measurement in {column}")
    data.index.name = "source_row"
    return data


def remove_incomplete_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep complete observations and their original source-row identifiers."""
    data = raw.dropna(subset=["species", *FEATURE_NAMES]).copy()
    if data.empty or set(data["species"].unique()) != set(SPECIES):
        raise ValueError("Complete Penguins rows must contain all three species")
    return data


def load_penguins_dataset(archive_path: Path = PENGUINS_ARCHIVE) -> pd.DataFrame:
    """Return complete cases only; all incomplete source rows are removed."""
    return remove_incomplete_rows(load_raw_penguins_dataset(archive_path))
