"""Canonical WDBC loading, validation and preprocessing."""

from __future__ import annotations

from io import TextIOWrapper
from zipfile import ZipFile

import pandas as pd

from .config import DATA_ARCHIVE


RANDOM_STATE = 42
DATA_MEMBER = "wdbc.data"
MEASURE_NAMES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
FEATURE_NAMES = [
    f"{stat}_{measure}"
    for stat in ("mean", "se", "worst")
    for measure in MEASURE_NAMES
]
COLUMN_NAMES = ["id", "diagnosis", *FEATURE_NAMES]
TARGET_MAP = {"B": 0, "M": 1}
TARGET_LABELS = {0: "Benign", 1: "Malignant"}


def load_raw_dataset() -> pd.DataFrame:
    """Read WDBC directly from the sole raw ZIP archive."""
    if not DATA_ARCHIVE.exists():
        raise FileNotFoundError(f"Dataset archive not found: {DATA_ARCHIVE}")

    with ZipFile(DATA_ARCHIVE) as archive:
        if DATA_MEMBER not in archive.namelist():
            raise ValueError(f"{DATA_ARCHIVE.name} does not contain {DATA_MEMBER}")
        with archive.open(DATA_MEMBER) as binary_file:
            data = pd.read_csv(
                TextIOWrapper(binary_file, encoding="utf-8"),
                header=None,
                names=COLUMN_NAMES,
            )

    if data.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns, received {data.shape[1]}"
        )
    unknown_targets = set(data["diagnosis"].dropna().unique()) - set(TARGET_MAP)
    if unknown_targets:
        raise ValueError(f"Unknown diagnosis values: {sorted(unknown_targets)}")
    return data


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply deterministic, medically conservative cleaning.

    Extreme measurements are flagged using IQR but are not clipped or removed:
    in a clinical dataset they may be valid and diagnostically important.
    """
    data = raw.copy()
    rows_before = len(data)
    missing_before = data.isna().sum()

    data = data.dropna(subset=["id", "diagnosis"]).copy()
    for column in FEATURE_NAMES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())

    duplicate_mask = data.duplicated(subset=["diagnosis", *FEATURE_NAMES])
    duplicate_count = int(duplicate_mask.sum())
    data = data.loc[~duplicate_mask].reset_index(drop=True)
    data["id"] = pd.to_numeric(data["id"], errors="raise").astype("int64")
    data["diagnosis_binary"] = data["diagnosis"].map(TARGET_MAP).astype("int8")

    outlier_counts: dict[str, int] = {}
    for column in FEATURE_NAMES:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (data[column] < q1 - 1.5 * iqr) | (data[column] > q3 + 1.5 * iqr)
        outlier_counts[column] = int(mask.sum())

    report: dict[str, object] = {
        "rows_before": rows_before,
        "rows_after": len(data),
        "duplicates_removed": duplicate_count,
        "missing_before": missing_before.to_dict(),
        "missing_after": data.isna().sum().to_dict(),
        "outlier_counts": outlier_counts,
    }
    return data, report


def load_clean_dataset() -> pd.DataFrame:
    """Return the validated, cleaned canonical analysis table."""
    data, _ = clean_dataset(load_raw_dataset())
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the 30 predictors and malignant-positive binary target."""
    return data[FEATURE_NAMES].copy(), data["diagnosis_binary"].copy()


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Return Tukey 1.5×IQR bounds."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def class_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Build a stable diagnosis count/share table."""
    counts = data["diagnosis"].value_counts().reindex(["B", "M"])
    return pd.DataFrame(
        {
            "diagnosis": ["B", "M"],
            "label": ["Benign", "Malignant"],
            "count": counts.to_numpy(),
            "share": (counts / counts.sum()).to_numpy(),
        }
    )
