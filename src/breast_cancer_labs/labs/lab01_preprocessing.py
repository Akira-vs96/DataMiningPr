"""Lab 01: data quality, outlier audit and standardization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from breast_cancer_labs.config import PROCESSED_DIR, prepare_output_dir
from breast_cancer_labs.data import (
    FEATURE_NAMES,
    class_distribution,
    clean_dataset,
    iqr_bounds,
    load_raw_dataset,
)
from breast_cancer_labs.plotting import COLORS, apply_style, save_figure


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab01_preprocessing")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    data, quality = clean_dataset(raw)
    distribution = class_distribution(data)

    clean_path = PROCESSED_DIR / "breast_cancer_clean.csv"
    scaled_path = PROCESSED_DIR / "breast_cancer_standardized.csv"
    data.to_csv(clean_path, index=False)

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(data[FEATURE_NAMES]),
        columns=[f"{column}_z" for column in FEATURE_NAMES],
    )
    standardized = pd.concat(
        [data[["id", "diagnosis", "diagnosis_binary"]].reset_index(drop=True), scaled],
        axis=1,
    )
    standardized.to_csv(scaled_path, index=False)

    summary = data[FEATURE_NAMES].describe().T
    summary.index.name = "feature"
    summary.to_csv(output_dir / "descriptive_statistics.csv")
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    outlier_table = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "lower_bound": [iqr_bounds(data[c])[0] for c in FEATURE_NAMES],
            "upper_bound": [iqr_bounds(data[c])[1] for c in FEATURE_NAMES],
            "outlier_count": [quality["outlier_counts"][c] for c in FEATURE_NAMES],
        }
    ).sort_values("outlier_count", ascending=False)
    outlier_table.to_csv(output_dir / "iqr_outlier_audit.csv", index=False)

    selected = ["mean_radius", "mean_texture", "mean_area", "mean_concavity"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, feature in zip(axes.flat, selected):
        values = [
            data.loc[data["diagnosis"] == diagnosis, feature]
            for diagnosis in ["B", "M"]
        ]
        boxes = axis.boxplot(values, tick_labels=["Benign", "Malignant"], patch_artist=True)
        for box, label in zip(boxes["boxes"], ["Benign", "Malignant"]):
            box.set_facecolor(COLORS[label])
            box.set_alpha(0.7)
        axis.set_title(feature)
        axis.set_ylabel("Original scale")
    fig.suptitle("Selected WDBC features and IQR-visible extremes")
    save_figure(fig, output_dir / "feature_boxplots.png")

    missing_total = int(sum(quality["missing_before"].values()))
    report = f"""# Лабораторная работа №1 — подготовка данных

## Результат

- Исходная форма: **{raw.shape[0]} × {raw.shape[1]}**.
- После очистки: **{data.shape[0]} × {data.shape[1]}** (включая бинарный target).
- Пропусков до обработки: **{missing_total}**.
- Полных дубликатов измерений удалено: **{quality['duplicates_removed']}**.
- Классы: benign — **{int(distribution.loc[0, 'count'])}**, malignant — **{int(distribution.loc[1, 'count'])}**.

## Принятые решения

ID не используется как признак. Диагноз закодирован как `B=0`, `M=1`.
Числовые признаки стандартизированы с помощью StandardScaler в отдельной таблице.
Наблюдения за границами 1.5×IQR отмечены в аудите, но не удалены и не обрезаны:
экстремальные клинические измерения могут быть корректными и важными для диагноза.

Очищенные данные сохранены в `{clean_path.relative_to(clean_path.parents[2])}`,
стандартизированные — в `{scaled_path.relative_to(scaled_path.parents[2])}`.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Rows: {len(data)}; features: {len(FEATURE_NAMES)}; missing: {missing_total}")
    print(f"Saved clean and standardized datasets to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
