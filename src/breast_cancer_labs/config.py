"""Project-wide paths and output helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "breast_cancer_wisconsin_diagnostic.zip"
PENGUINS_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "archive.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def prepare_output_dir(name: str) -> Path:
    """Create a lab output directory and remove its stale generated files."""
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    return output_dir
