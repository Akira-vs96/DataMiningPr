"""Run every implemented laboratory work in a deterministic order."""

from __future__ import annotations

import argparse
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
    lab09_ensemble_learning,
    lab10_time_series,
    lab11_explainable_ai,
    lab12_concept_drift,
    lab13_graph_analysis,
    lab14_topic_modeling,
    lab15_prediction_intervals,
)


LABS = [
    ("01", "Preprocessing", lab01_preprocessing.main),
    ("02", "Correlation", lab02_correlation.main),
    ("03", "Linear regression", lab03_regression.main),
    ("04", "Hypothesis testing", lab04_hypothesis_testing.main),
    ("05", "Clustering", lab05_clustering.main),
    ("07", "Association rules", lab07_association_rules.main),
    ("08", "Gradient Boosting and XGBoost", lab08_boosting.main),
    ("09", "Penguins: Stacking and Blending (variant 1)", lab09_ensemble_learning.main),
    ("10", "Electricity: SARIMA (variant 1)", lab10_time_series.main),
    ("11", "Wine: SHAP and LIME (variant 1)", lab11_explainable_ai.main),
    ("12", "Electricity: concept drift (variant 1)", lab12_concept_drift.main),
    ("13", "Karate Club: graph analysis (variant 1)", lab13_graph_analysis.main),
    ("14", "Scientific abstracts: LDA and NMF (variant 1)", lab14_topic_modeling.main),
    ("15", "Electricity: conformal intervals (variant 1)", lab15_prediction_intervals.main),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", nargs="+", choices=[number for number, _, _ in LABS],
                        help="Run only the selected laboratory numbers, e.g. --labs 10 11 12 13 14 15")
    args = parser.parse_args()
    selected = [lab for lab in LABS if args.labs is None or lab[0] in args.labs]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, name, function in selected:
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
    summary_path = RESULTS_DIR / "run_summary.csv"
    if args.labs and summary_path.exists():
        previous = pd.read_csv(summary_path, dtype={"lab": str})
        previous = previous[~previous.lab.isin(summary.lab)]
        pd.concat([previous, summary], ignore_index=True).sort_values("lab").to_csv(summary_path, index=False)
    else:
        summary.to_csv(summary_path, index=False)
    report_links = "\n".join(
        f"- Лабораторная №{number}: [{name}]({function.__module__.rsplit('.', 1)[-1]}/report.md)"
        for number, name, function in LABS
    )
    index = f"""# Результаты лабораторных работ

Результаты генерируются командой `.venv/bin/python run_all.py`.
Только варианты 1 практик 10–15: `.venv/bin/python run_all.py --labs 10 11 12 13 14 15`.
Лабораторные №1–5, №7 и №8 используют Breast Cancer Wisconsin Diagnostic;
практическая №9 (вариант 1) — Penguins из `data/raw/archive.zip`.
Практики 10–15 содержат только вариант 1: электропотребление, Wine,
контролируемый drift, Karate Club, научные аннотации и прогнозные интервалы.

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
