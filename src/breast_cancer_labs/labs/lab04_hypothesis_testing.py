"""Lab 04: compare mean radius between benign and malignant diagnoses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.data import RANDOM_STATE, load_clean_dataset
from breast_cancer_labs.plotting import COLORS, apply_style, save_figure


FEATURE = "mean_radius"


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    pooled_variance = (
        (len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)
    ) / (len(first) + len(second) - 2)
    cohen_d = (second.mean() - first.mean()) / np.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (len(first) + len(second)) - 9)
    return float(cohen_d * correction)


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab04_hypothesis_testing")
    data = load_clean_dataset()
    benign = data.loc[data["diagnosis"] == "B", FEATURE].to_numpy()
    malignant = data.loc[data["diagnosis"] == "M", FEATURE].to_numpy()

    summary = (
        data.assign(label=data["diagnosis"].map({"B": "Benign", "M": "Malignant"}))
        .groupby("label")[FEATURE]
        .agg(count="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    summary.to_csv(output_dir / "group_summary.csv", index=False)

    shapiro_b = stats.shapiro(benign)
    shapiro_m = stats.shapiro(malignant)
    levene = stats.levene(benign, malignant, center="median")
    welch = stats.ttest_ind(benign, malignant, equal_var=False)
    mann_whitney = stats.mannwhitneyu(benign, malignant, alternative="two-sided")
    effect = hedges_g(benign, malignant)
    tests = pd.DataFrame(
        [
            ("Shapiro-Wilk: Benign", shapiro_b.statistic, shapiro_b.pvalue),
            ("Shapiro-Wilk: Malignant", shapiro_m.statistic, shapiro_m.pvalue),
            ("Levene", levene.statistic, levene.pvalue),
            ("Welch t-test", welch.statistic, welch.pvalue),
            ("Mann-Whitney U", mann_whitney.statistic, mann_whitney.pvalue),
        ],
        columns=["test", "statistic", "p_value"],
    )
    tests.to_csv(output_dir / "hypothesis_tests.csv", index=False)
    pd.DataFrame([{"effect": "Hedges g", "value": effect}]).to_csv(
        output_dir / "effect_size.csv", index=False
    )

    rng = np.random.default_rng(RANDOM_STATE)
    fig, axis = plt.subplots(figsize=(8, 6))
    groups = [benign, malignant]
    labels = ["Benign", "Malignant"]
    boxes = axis.boxplot(groups, tick_labels=labels, widths=0.5, patch_artist=True)
    for position, (values, label, box) in enumerate(zip(groups, labels, boxes["boxes"]), start=1):
        box.set_facecolor(COLORS[label])
        box.set_alpha(0.5)
        axis.scatter(
            rng.normal(position, 0.045, len(values)),
            values,
            s=10,
            alpha=0.28,
            color=COLORS[label],
        )
    axis.set(ylabel=FEATURE, title="Mean radius by diagnosis")
    save_figure(fig, output_dir / "mean_radius_by_diagnosis.png")

    decision = "отвергается" if welch.pvalue < 0.05 else "не отвергается"
    report = f"""# Лабораторная работа №4 — проверка гипотез

H₀: средние значения `{FEATURE}` в benign- и malignant-группах равны.
H₁: средние значения различаются.

Welch t-test: t = **{welch.statistic:.4f}**, p = **{welch.pvalue:.3e}**;
при α=0.05 нулевая гипотеза **{decision}**. Непараметрический Mann–Whitney
также рассчитан как проверка устойчивости вывода (p = {mann_whitney.pvalue:.3e}).

Размер эффекта Hedges g = **{effect:.4f}** (положительный знак означает большее
значение в malignant-группе). Welch-вариант выбран заранее, поскольку он не
требует равенства дисперсий.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(tests.to_string(index=False))
    print(f"Hedges g: {effect:.4f}")


if __name__ == "__main__":
    main()
