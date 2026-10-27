"""Practical 15, variant 1: conformal intervals for daily electricity demand."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.plotting import apply_style, save_figure
from breast_cancer_labs.practical_common import (
    RAW_DIR, electricity_features, load_daily_electricity, markdown_table, regression_metrics, save_environment,
)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Need finite nonnegative residuals and 0 < alpha < 1")
    k = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[k - 1]) if k <= len(values) else float("inf")


def season(month: int) -> str:
    # Victoria is in the southern hemisphere.
    return {12: "summer", 1: "summer", 2: "summer", 3: "autumn", 4: "autumn", 5: "autumn",
            6: "winter", 7: "winter", 8: "winter", 9: "spring", 10: "spring", 11: "spring"}[month]


def main() -> None:
    apply_style()
    output = prepare_output_dir("lab15_prediction_intervals")
    data = load_daily_electricity()
    X, y = electricity_features(data), data.Demand
    n_train, n_cal = int(.60 * len(data)), int(.20 * len(data))
    train_end, cal_end = n_train, n_train + n_cal
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X.iloc[:train_end], y.iloc[:train_end])
    cal_prediction = model.predict(X.iloc[train_end:cal_end])
    actual = y.iloc[cal_end:]
    prediction = model.predict(X.iloc[cal_end:])
    residuals = np.abs(y.iloc[train_end:cal_end].to_numpy() - cal_prediction)
    scores = pd.DataFrame([regression_metrics(actual, prediction)])
    intervals = pd.DataFrame({"actual": actual, "prediction": prediction})
    summary, grouped = [], []
    for alpha in [.20, .10, .05]:
        target = int(round((1 - alpha) * 100))
        q = conformal_quantile(residuals, alpha)
        lower, upper = prediction - q, prediction + q
        inside = (actual.to_numpy() >= lower) & (actual.to_numpy() <= upper)
        intervals[f"lower_{target}"] = lower
        intervals[f"upper_{target}"] = upper
        intervals[f"inside_{target}"] = inside
        summary.append({"target_coverage": 1 - alpha, "calibration_n": n_cal,
                        "order_statistic_k": math.ceil((n_cal + 1) * (1 - alpha)), "q": q,
                        "test_n": len(actual), "covered": int(inside.sum()),
                        "empirical_coverage": float(inside.mean()), "mean_width": float(np.mean(upper - lower))})
        labels = pd.Series([season(m) for m in actual.index.month], index=actual.index)
        for group in labels.unique():
            mask = labels.eq(group).to_numpy()
            grouped.append({"season": group, "target_coverage": 1 - alpha, "n": int(mask.sum()),
                            "coverage": float(inside[mask].mean()), "mean_width": 2 * q,
                            "mae": float(np.abs(actual.to_numpy()[mask] - prediction[mask]).mean())})
    summary, grouped = pd.DataFrame(summary), pd.DataFrame(grouped)
    splits = pd.DataFrame({"split": ["train"] * n_train + ["calibration"] * n_cal + ["test"] * len(actual), "actual": y})
    splits.to_csv(output / "data_split.csv", index_label="date")
    intervals.to_csv(output / "test_intervals.csv", index_label="date")
    scores.to_csv(output / "point_metrics.csv", index=False)
    summary.to_csv(output / "interval_comparison.csv", index=False)
    grouped.to_csv(output / "seasonal_coverage.csv", index=False)
    pd.DataFrame({"actual": y.iloc[train_end:cal_end], "prediction": cal_prediction,
                  "absolute_residual": residuals}).to_csv(output / "calibration_residuals.csv", index_label="date")
    X.to_csv(output / "features.csv", index_label="date")
    pd.DataFrame({"feature": X.columns, "coefficient_scaled": model.named_steps["ridge"].coef_}).to_csv(output / "model_coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(actual.index, intervals.lower_95, intervals.upper_95, alpha=.15, color="blue", label="95%")
    ax.fill_between(actual.index, intervals.lower_90, intervals.upper_90, alpha=.2, color="blue", label="90%")
    ax.fill_between(actual.index, intervals.lower_80, intervals.upper_80, alpha=.25, color="blue", label="80%")
    ax.plot(actual.index, actual, "o", markersize=3, color="black", label="Actual")
    ax.plot(actual.index, prediction, color="orange", label="Ridge prediction")
    ax.set(ylabel="Source daily-demand units", title="Chronological test: daily electricity prediction intervals")
    ax.legend(ncol=5)
    save_figure(fig, output / "prediction_intervals.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(residuals, bins=18, alpha=.7)
    for row in summary.itertuples():
        axes[0].axvline(row.q, label=f"{row.target_coverage:.0%}: q={row.q:.1f}")
    axes[0].legend()
    axes[0].set(title="Calibration absolute residuals", xlabel="Absolute residual")
    axes[1].plot(summary.target_coverage, summary.empirical_coverage, "o-", label="Test coverage")
    axes[1].plot([.75, 1], [.75, 1], "--", label="Target")
    axes[1].set(xlabel="Target coverage", ylabel="Empirical test coverage", ylim=(0, 1.05))
    axes[1].legend()
    save_figure(fig, output / "calibration_coverage.png")
    fig, ax = plt.subplots(figsize=(12, 4))
    for group, part in splits.groupby("split", sort=False):
        ax.plot(part.index, part.actual, label=group)
    ax.set(title="Daily electricity demand and chronological splits", ylabel="Source daily-demand units")
    ax.legend()
    save_figure(fig, output / "chronological_split.png")
    report = f"""# Практическая работа №15 — вариант 1

Тема: неопределённость прогнозов суточного электропотребления.
[Задание](../../Data_Mining_Practica_15.docx).

## Источник и признаки

Реальные 365 дневных наблюдений Victoria, Australia, за 2014 год: fpp2 **elecdaily**,
[CSV](../../data/raw/elecdaily.csv), [документация](https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdaily.html).
Первичные источники — AEMO и Australian Bureau of Meteorology.
Цель — исходный агрегированный показатель Demand. Документация fpp2 обозначает его GW даже
для дневной агрегации; поэтому здесь сохранены исходные единицы и **не заявлен пересчёт в кВт·ч**.
Temperature — дневной максимум температуры в °C, WorkDay — рабочий день (0/1).
Даты восстановлены как последовательные дни 2014 года согласно описанию источника;
пропусков нет, исходные значения не изменены.

Признаки: температура, её квадрат, WorkDay, sin/cos годового и недельного календарного цикла.
Наблюдаемая температура служит известной ковариатой: для реального прогноза заранее
понадобится её прогноз, а его дополнительная неопределённость здесь не учтена.

## Хронологическое разбиение и модель

Train — {n_train} дней ({y.index[0]:%Y-%m-%d}–{y.index[train_end-1]:%Y-%m-%d}),
calibration — {n_cal} ({y.index[train_end]:%Y-%m-%d}–{y.index[cal_end-1]:%Y-%m-%d}),
test — {len(actual)} ({actual.index[0]:%Y-%m-%d}–{actual.index[-1]:%Y-%m-%d}).
Перемешивания нет. StandardScaler + Ridge(alpha=10) обучены только на train;
calibration задаёт квантили, test используется только для итоговой оценки.
Модель не переобучается на calibration после выбора квантилей.

![Разбиение](chronological_split.png)

## Наблюдаемые результаты

{markdown_table(scores)}

Для каждого α использован **k = ceil((n_cal+1)(1−α))** и k-й абсолютный
калибровочный остаток q. Если k > n_cal, конечной границы нет: q=∞.
Индивидуальный прогнозный интервал равен [ŷ−q, ŷ+q].

{markdown_table(summary)}

![Прогнозные интервалы](prediction_intervals.png)
![Калибровка и покрытие](calibration_coverage.png)

Чем выше целевой уровень покрытия, тем больше или равно q и ширина вложенных интервалов.
MAE/RMSE и ширина имеют единицы Demand, R² и покрытие безразмерны.
Это интервалы **для новых индивидуальных наблюдений**, а доверительный интервал среднего
описывал бы неопределённость среднего уровня и решал другую задачу.

## Сезонность и ограничения

В южном полушарии декабрь–февраль — лето, сентябрь–ноябрь — весна.
Годовой график позволяет видеть недельные колебания и сезонные изменения;
одного года недостаточно для проверки повторяемости годового цикла.
Test охватывает лишь часть весны и начало лета, поэтому сравнение всех четырёх сезонов невозможно.

{markdown_table(grouped)}

Для обменных наблюдений split conformal даёт конечновыборочное **маргинальное** покрытие,
но не точное покрытие каждой точки или группы. Последовательные дневные данные зависимы,
а train/calibration/test относятся к разным сезонам: обменность здесь не подтверждена.
Поэтому 80/90/95% — номинальные уровни, формальную гарантию для этого ряда утверждать нельзя.
Покрытие и ширина выше — фактически измеренные значения на одном test-периоде;
простая постоянная ширина не адаптируется к изменению дисперсии ошибки.

## Вывод

Обучена базовая регрессия суточного потребления с погодными и календарными признаками.
Три интервала рассчитаны по отдельным калибровочным остаткам и проверены на будущих днях.
Увеличение номинального покрытия расширяет интервалы, а фактическое покрытие зависит от сезона.
Для применения к временному ряду нужна дополнительная проверка зависимости и устойчивости к сдвигу распределения.

Запуск: `PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab15_prediction_intervals`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    save_environment(output, [RAW_DIR / "elecdaily.csv"], train_n=n_train, calibration_n=n_cal, test_n=len(actual),
                     ridge_alpha=10, interval_targets=[.8, .9, .95], exchangeability_guarantee_claimed=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
