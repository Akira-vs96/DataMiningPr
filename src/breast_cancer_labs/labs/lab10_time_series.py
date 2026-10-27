"""Practical 10, variant 1: monthly electricity consumption forecasting."""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.plotting import apply_style, save_figure
from breast_cancer_labs.practical_common import RAW_DIR, markdown_table, regression_metrics, save_environment


def seasonal_naive(train: pd.Series, horizon: int, period: int = 12) -> np.ndarray:
    if len(train) < period or horizon < 1:
        raise ValueError("Need one full season and a positive horizon")
    return np.resize(train.iloc[-period:].to_numpy(), horizon)


def fit_sarima(train, order):
    return SARIMAX(train, order=order, seasonal_order=(0, 1, 1, 12),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)


def main() -> None:
    apply_style()
    output = prepare_output_dir("lab10_time_series")
    payload = json.loads((RAW_DIR / "electricity_monthly_eia.json").read_text())
    raw = pd.DataFrame(payload["response"]["data"]).sort_values("period")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.period), name="date")
    values = pd.to_numeric(raw.sales, errors="raise").to_numpy()
    if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="MS", name="date")) or not np.isfinite(values).all():
        raise ValueError("Monthly input has missing periods, duplicates or missing values")
    if set(raw["sales-units"]) != {"million kilowatt hours"} or set(raw.stateid) != {"US"} or set(raw.sectorid) != {"ALL"}:
        raise ValueError("Unexpected EIA geography, sector or unit")
    series = pd.Series(values, index=pd.date_range(dates.min(), periods=len(raw), freq="MS"), name="consumption_gwh")
    train, test = series.iloc[:-24], series.iloc[-24:]
    inner, valid = train.iloc[:-24], train.iloc[-24:]
    candidates, fits = [], {}
    for order in [(0, 1, 1), (1, 1, 0), (1, 1, 1)]:
        with warnings.catch_warnings(record=True) as caught:
            fitted = fit_sarima(inner, order)
        converged = bool(fitted.mle_retvals.get("converged", False))
        candidates.append({"order": str(order), "seasonal_order": "(0,1,1,12)",
                           "validation_rmse": regression_metrics(valid, fitted.forecast(len(valid)))["rmse"],
                           "converged": converged, "warnings": " / ".join(str(w.message) for w in caught)})
        fits[str(order)] = order
    tuning = pd.DataFrame(candidates)
    eligible = tuning[tuning.converged]
    if eligible.empty:
        raise RuntimeError("No SARIMA candidate converged")
    selected = fits[eligible.sort_values("validation_rmse").iloc[0].order]
    model = fit_sarima(train, selected)
    if not model.mle_retvals.get("converged", False):
        raise RuntimeError("Final SARIMA did not converge")
    forecast = model.get_forecast(len(test))
    interval = forecast.conf_int()
    predictions = pd.DataFrame({"actual": test, "seasonal_naive": seasonal_naive(train, len(test)),
                                "sarima": forecast.predicted_mean,
                                "lower_95": interval.iloc[:, 0], "upper_95": interval.iloc[:, 1]})
    metrics = pd.DataFrame([{"model": name, **regression_metrics(test, predictions[column]),
                             "mape_percent": np.mean(np.abs((test - predictions[column]) / test)) * 100}
                            for name, column in [("Seasonal naive", "seasonal_naive"), ("SARIMA", "sarima")]])
    adf = pd.DataFrame([{"series": name, "statistic": result[0], "p_value": result[1], "lags": result[2], "nobs": result[3]}
                        for name, values in [("train", train), ("first_difference", train.diff().dropna()),
                                             ("seasonal_and_first_difference", train.diff(12).diff().dropna())]
                        for result in [adfuller(values)]])
    decomposition = seasonal_decompose(train, model="additive", period=12)
    components = pd.DataFrame({"observed": train, "trend": decomposition.trend,
                               "seasonal": decomposition.seasonal, "residual": decomposition.resid})
    monthly = components.groupby(components.index.month).seasonal.mean().rename_axis("month").reset_index(name="seasonal_gwh")
    series.to_csv(output / "monthly_consumption.csv", index_label="date")
    pd.DataFrame({"value": series, "split": ["train"] * len(train) + ["test"] * len(test)}).to_csv(output / "data_split.csv", index_label="date")
    for name, table in [("model_comparison", metrics), ("sarima_validation", tuning), ("adf", adf), ("seasonality_by_month", monthly)]:
        table.to_csv(output / f"{name}.csv", index=False)
    predictions.to_csv(output / "forecast.csv", index_label="date")
    components.to_csv(output / "decomposition.csv", index_label="date")
    (output / "sarima_summary.txt").write_text(str(model.summary()), encoding="utf-8")
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    save_figure(fig, output / "decomposition.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train.diff(12).diff().dropna(), lags=36, ax=axes[0])
    plot_pacf(train.diff(12).diff().dropna(), lags=36, method="ywm", ax=axes[1])
    save_figure(fig, output / "acf_pacf.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    train.iloc[-72:].plot(ax=ax, label="Train")
    for column in ["actual", "seasonal_naive", "sarima"]:
        predictions[column].plot(ax=ax, label=column)
    ax.fill_between(test.index, predictions.lower_95, predictions.upper_95, alpha=.2, label="SARIMA 95% prediction interval")
    ax.set(ylabel="GWh", title="US monthly electricity consumption: fixed 24-month forecast")
    ax.legend()
    save_figure(fig, output / "forecast.png")
    peak = int(monthly.loc[monthly.seasonal_gwh.idxmax(), "month"])
    winner = metrics.sort_values("rmse").iloc[0]
    report = f"""# Практическая работа №10 — вариант 1

Тема: интеллектуальный анализ и прогноз месячного электропотребления.
Задание: [методичка](../../Data_Mining_Practica_10.docx).

## Данные и подготовка

Реальный открытый ряд **EIA Electricity Retail Sales**, US, ALL: месячные продажи электроэнергии
конечным потребителям США (операционное измерение потребления),
{series.index.min():%Y-%m}–{series.index.max():%Y-%m}, {len(series)} наблюдений, единица — GWh.
[JSON API](../../data/raw/electricity_monthly_eia.json), [источник EIA](https://www.eia.gov/electricity/data/browser/).
В кВт·ч значение переводится умножением на 1 000 000. Пропусков, повторных и отсутствующих месяцев нет;
интерполяция не применялась. Train: {train.index.min():%Y-%m}–{train.index.max():%Y-%m} ({len(train)} месяцев),
test: {test.index.min():%Y-%m}–{test.index.max():%Y-%m} (24 месяца). Перемешивания нет.

## Тренд, сезонность, стационарность

Среднее первых 12 train-месяцев — {train.iloc[:12].mean():.1f} GWh,
последних 12 — {train.iloc[-12:].mean():.1f} GWh: долгосрочный уровень вырос.
Аддитивная декомпозиция только train показывает 12-месячный цикл;
**сезонный максимум приходится на месяц {peak}** (1 — январь, 12 — декабрь).
Сравнивается сезонная компонента, чтобы тренд не смещал средние по месяцам.

{markdown_table(adf)}

ADF проверяет нулевую гипотезу единичного корня; p < 0.05 позволяет её отвергнуть,
но не доказывает все свойства стационарности. ACF/PACF построены после сезонной и первой разности.

![Декомпозиция train](decomposition.png)
![ACF и PACF](acf_pacf.png)

## Модели и честная оценка

Последние 24 месяца **внутри train** использованы для выбора параметров по RMSE.
Проверены три SARIMA с сезонным порядком (0,1,1,12); выбран порядок {selected}.
Финальная модель обучена на полном train. Test не участвует в выборе параметров.
Baseline циклически повторяет последние 12 значений train; обе модели одновременно
прогнозируют 24 месяца от одной даты и не получают истинных значений test.

{markdown_table(tuning.drop(columns='warnings'))}

{markdown_table(metrics)}

MAE и RMSE измеряются в GWh, MAPE — в процентах; все фактические значения положительны.
Наименьший RMSE здесь у **{winner.model}**: {winner.rmse:.2f} GWh.
95% интервал SARIMA относится к новым наблюдениям при предположениях модели;
это не гарантия и не интервал среднего потребления.

![Прогноз](forecast.png)

## Вывод

В месячном потреблении обнаружены рост уровня и повторение сезонного цикла.
Сезонный пик определён на обучающем периоде: месяц {peak}.
Сравнение на одном фиксированном горизонте показало меньший RMSE у {winner.model}.
Результат относится к одному историческому test-периоду и не гарантирует такую же точность в будущем.

Запуск: `PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab10_time_series`.
Таблицы проверки, прогнозы, параметры, сводка SARIMA и версии среды сохранены рядом с отчётом.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    save_environment(output, [RAW_DIR / "electricity_monthly_eia.json"], horizon=24, selected_order=selected, seasonal_order=[0, 1, 1, 12])
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
