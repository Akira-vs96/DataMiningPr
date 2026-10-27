"""Load the supplied Penguins archive without learning preprocessing statistics."""

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


def load_penguins_dataset(archive_path: Path = PENGUINS_ARCHIVE) -> pd.DataFrame:
    """Validate schema and labels, keeping missing predictors for fold-local imputation.

    No rows are dropped and no values are imputed here. The stable source-row
    index allows every train/validation/test assignment to be audited later.
    """
    with ZipFile(archive_path) as archive:
        with archive.open(DATA_MEMBER) as source:
            data = pd.read_csv(source)

    expected = ["species", *FEATURE_NAMES]
    if set(data.columns) != set(expected):
        raise ValueError(f"Expected columns {expected}, received {data.columns.tolist()}")
    data = data[expected].copy()
    if data.empty or data["species"].isna().any():
        raise ValueError("Penguins must have non-empty, complete species labels")
    if set(data["species"].unique()) != set(SPECIES):
        raise ValueError(f"Expected species {SPECIES}")
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(f"Infinite measurement in {column}")
    data.index.name = "source_row"
    return data
