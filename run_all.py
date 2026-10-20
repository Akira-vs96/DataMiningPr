"""Run every implemented laboratory work in a deterministic order."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from breast_cancer_labs.config import RESULTS_DIR  # noqa: E402
from breast_cancer_labs.labs import (  # noqa: E402
    lab01_preprocessing,
    lab02_correlation,
    lab03_regression,
    lab04_hypothesis_testing,
    lab05_clustering,
    lab07_association_rules,
    lab08_boosting,
)


LABS = [
    ("01", "Preprocessing", lab01_preprocessing.main),
    ("02", "Correlation", lab02_correlation.main),
    ("03", "Linear regression", lab03_regression.main),
    ("04", "Hypothesis testing", lab04_hypothesis_testing.main),
    ("05", "Clustering", lab05_clustering.main),
    ("07", "Association rules", lab07_association_rules.main),
    ("08", "Gradient Boosting and XGBoost", lab08_boosting.main),
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, name, function in LABS:
        print(f"\n{'=' * 72}\nLab {number}: {name}\n{'=' * 72}")
        started = time.perf_counter()
        function()
        rows.append(
            {
                "lab": number,
                "name": name,
                "status": "success",
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "run_summary.csv", index=False)
    report_links = "\n".join(
        f"- Лабораторная №{number}: [{name}]({function.__module__.rsplit('.', 1)[-1]}/report.md)"
        for number, name, function in LABS
    )
    index = f"""# Результаты лабораторных работ

Все результаты сгенерированы командой `.venv/bin/python run_all.py` на едином
датасете Breast Cancer Wisconsin Diagnostic.

{report_links}

Машиночитаемый статус и время выполнения находятся в `run_summary.csv`.
Каждый каталог содержит собственные таблицы CSV, графики PNG и краткий
аналитический отчёт `report.md`.
"""
    (RESULTS_DIR / "README.md").write_text(index, encoding="utf-8")
    print("\nAll labs completed successfully.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
