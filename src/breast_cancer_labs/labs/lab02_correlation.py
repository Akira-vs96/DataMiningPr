"""Lab 02: correlation analysis on WDBC features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, pointbiserialr, spearmanr

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.data import FEATURE_NAMES, load_clean_dataset
from breast_cancer_labs.plotting import COLORS, apply_style, save_figure


X_FEATURE = "mean_radius"
Y_FEATURE = "worst_radius"
HEATMAP_FEATURES = [feature for feature in FEATURE_NAMES if feature.startswith("mean_")]


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab02_correlation")
    data = load_clean_dataset()
    x, y = data[X_FEATURE], data[Y_FEATURE]

    tests = [
        ("Pearson", *pearsonr(x, y)),
        ("Spearman", *spearmanr(x, y)),
        ("Kendall", *kendalltau(x, y)),
    ]
    pair_results = pd.DataFrame(tests, columns=["method", "coefficient", "p_value"])
    pair_results.to_csv(output_dir / "pair_correlation.csv", index=False)

    target_rows = []
    for feature in FEATURE_NAMES:
        coefficient, p_value = pointbiserialr(data["diagnosis_binary"], data[feature])
        target_rows.append(
            {"feature": feature, "point_biserial_r": coefficient, "p_value": p_value}
        )
    target_results = pd.DataFrame(target_rows)
    target_results["absolute_r"] = target_results["point_biserial_r"].abs()
    target_results = target_results.sort_values("absolute_r", ascending=False)
    target_results.to_csv(output_dir / "feature_target_correlations.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 6))
    for diagnosis, label in [("B", "Benign"), ("M", "Malignant")]:
        subset = data[data["diagnosis"] == diagnosis]
        axis.scatter(
            subset[X_FEATURE],
            subset[Y_FEATURE],
            s=28,
            alpha=0.65,
            label=label,
            color=COLORS[label],
        )
    axis.set(xlabel=X_FEATURE, ylabel=Y_FEATURE, title="Mean radius vs worst radius")
    axis.legend()
    save_figure(fig, output_dir / "radius_scatter.png")

    matrix = data[HEATMAP_FEATURES].corr(method="pearson")
    matrix.to_csv(output_dir / "mean_features_correlation_matrix.csv")
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    short_labels = [name.removeprefix("mean_") for name in HEATMAP_FEATURES]
    axis.set_xticks(range(len(short_labels)), short_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(short_labels)), short_labels)
    for row in range(len(short_labels)):
        for column in range(len(short_labels)):
            axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    axis.set_title("Pearson correlation — mean features")
    fig.colorbar(image, ax=axis, shrink=0.8)
    save_figure(fig, output_dir / "correlation_heatmap.png")

    top = target_results.head(10).sort_values("point_biserial_r")
    fig, axis = plt.subplots(figsize=(9, 6))
    colors = np.where(top["point_biserial_r"] >= 0, "#e76f51", "#2a9d8f")
    axis.barh(top["feature"], top["point_biserial_r"], color=colors)
    axis.set(xlabel="Point-biserial r", title="Top-10 associations with malignant diagnosis")
    save_figure(fig, output_dir / "target_correlation_top10.png")

    strongest = target_results.iloc[0]
    pearson_value = pair_results.loc[pair_results["method"] == "Pearson", "coefficient"].iloc[0]
    report = f"""# Лабораторная работа №2 — корреляционный анализ

Связь `{X_FEATURE}` и `{Y_FEATURE}` сильная: Pearson r = **{pearson_value:.4f}**.
Также рассчитаны ранговые коэффициенты Spearman и Kendall, поэтому вывод не
зависит только от предположения о линейности.

Самая сильная одномерная связь с malignant-диагнозом обнаружена у
`{strongest['feature']}`: point-biserial r = **{strongest['point_biserial_r']:.4f}**
(p = {strongest['p_value']:.3e}). Знак показывает направление связи, но
корреляция не доказывает причинность и не заменяет оценку классификатора.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(pair_results.round(5).to_string(index=False))
    print(f"Strongest target association: {strongest['feature']}")


if __name__ == "__main__":
    main()
