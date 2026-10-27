"""Practical 12, variant 1: electricity regression and controlled concept drift."""

from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import drift
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.plotting import apply_style, save_figure
from breast_cancer_labs.practical_common import (
    RAW_DIR, electricity_features, load_daily_electricity, markdown_table, save_environment,
)

WARMUP = 60
CHANGE = 200
HORIZON = 100
CONFIGS = {
    "sensitive": {"adwin_delta": .01, "ph_threshold": 5.0},
    "conservative": {"adwin_delta": .002, "ph_threshold": 10.0},
}


def make_detectors(config: dict) -> dict:
    return {"ADWIN": drift.ADWIN(delta=config["adwin_delta"]),
            "Page-Hinkley": drift.PageHinkley(min_instances=30, delta=.005,
                                              threshold=config["ph_threshold"], mode="up")}


def alarm_summary(alarms, change=CHANGE, horizon=HORIZON) -> dict:
    timely = [a for a in alarms if change <= a <= change + horizon]
    return {"alarms": len(alarms), "false_before_change": sum(a < change for a in alarms),
            "first_timely_alarm": timely[0] if timely else None,
            "delay": timely[0] - change if timely else None,
            "late_alarms": sum(a > change + horizon for a in alarms), "timely_detected": bool(timely)}


def new_model(X, y) -> SGDRegressor:
    return SGDRegressor(loss="squared_error", learning_rate="constant", eta0=.003,
                        penalty="l2", alpha=.001, max_iter=3000, tol=1e-5,
                        shuffle=False, random_state=42).fit(X, y)


def main() -> None:
    apply_style()
    output = prepare_output_dir("lab12_concept_drift")
    data = load_daily_electricity()
    # A 60-day warmup cannot identify a yearly cycle reliably. Avoid extrapolating it.
    X = electricity_features(data).drop(columns=["annual_sin", "annual_cos"])
    original = data.Demand.to_numpy()
    # This is a known intervention for evaluation, NOT a claim about a real tariff event.
    target = original.copy()
    target[CHANGE:] *= .75
    scaler = StandardScaler().fit(X.iloc[:WARMUP])
    transformed = scaler.transform(X)
    y_mean, y_scale = float(target[:WARMUP].mean()), float(target[:WARMUP].std())
    normalized = (target - y_mean) / y_scale
    rows, alarm_rows = [], []
    for strategy in ["frozen", "online", "reset_adwin"]:
        model = new_model(transformed[:WARMUP], normalized[:WARMUP])
        detectors = {name: make_detectors(config) for name, config in CONFIGS.items()}
        recent = deque(range(WARMUP), maxlen=60)
        for i in range(WARMUP, len(data)):
            # Prediction and detector update precede all model updates at time i.
            prediction = float(model.predict(transformed[i:i+1])[0] * y_scale + y_mean)
            error = abs(target[i] - prediction)
            # ADWIN consumes a bounded signal; use a scale learned only in warmup.
            signal = error / (error + y_scale)
            alarms_now = []
            for config_name, pair in detectors.items():
                for detector_name, detector in pair.items():
                    detector.update(signal)
                    if detector.drift_detected:
                        alarms_now.append((config_name, detector_name))
                        alarm_rows.append({"strategy": strategy, "config": config_name,
                                           "detector": detector_name, "time": i, "date": str(data.index[i].date())})
            reset = strategy == "reset_adwin" and ("sensitive", "ADWIN") in alarms_now
            rows.append({"strategy": strategy, "time": i, "date": data.index[i],
                         "actual": target[i], "prediction": prediction,
                         "absolute_error": error, "detector_signal": signal, "reset_after_prediction": reset})
            recent.append(i)
            if reset:
                positions = list(recent)
                model = new_model(transformed[positions], normalized[positions])
            elif strategy == "online":
                model.partial_fit(transformed[i:i+1], normalized[i:i+1])
    history = pd.DataFrame(rows)
    history["rolling_mae_40"] = history.groupby("strategy").absolute_error.transform(lambda s: s.rolling(40, min_periods=10).mean())
    alarms = pd.DataFrame(alarm_rows, columns=["strategy", "config", "detector", "time", "date"])
    summaries = []
    for strategy in history.strategy.unique():
        for config_name, config in CONFIGS.items():
            for detector in ["ADWIN", "Page-Hinkley"]:
                subset = alarms[(alarms.strategy == strategy) & (alarms.config == config_name) & (alarms.detector == detector)]
                summaries.append({"strategy": strategy, "config": config_name, "detector": detector,
                                  **config, **alarm_summary(subset.time.tolist())})
    summaries = pd.DataFrame(summaries)
    periods = []
    for strategy, group in history.groupby("strategy", sort=False):
        for name, lower, upper in [("before_change", WARMUP, CHANGE), ("first_40_after", CHANGE, CHANGE + 40),
                                    ("late_after", CHANGE + 40, len(data))]:
            part = group[(group.time >= lower) & (group.time < upper)]
            periods.append({"strategy": strategy, "period": name, "n": len(part),
                            "mae": part.absolute_error.mean(), "rmse": np.sqrt(np.mean((part.actual-part.prediction)**2))})
    periods = pd.DataFrame(periods)
    windows = []
    for alarm in alarms.itertuples():
        part = history[history.strategy == alarm.strategy]
        before = part[(part.time >= alarm.time - 20) & (part.time < alarm.time)]
        after = part[(part.time > alarm.time) & (part.time <= alarm.time + 20)]
        windows.append({"strategy": alarm.strategy, "config": alarm.config, "detector": alarm.detector,
                        "alarm_time": alarm.time, "before_n": len(before), "after_n": len(after),
                        "mae_20_before": before.absolute_error.mean(), "mae_20_after": after.absolute_error.mean()})
    windows = pd.DataFrame(windows, columns=["strategy", "config", "detector", "alarm_time", "before_n", "after_n", "mae_20_before", "mae_20_after"])
    source = data.assign(time=np.arange(len(data)), original_demand=original, scenario_demand=target,
                         injected_change=np.arange(len(data)) >= CHANGE)
    source.to_csv(output / "controlled_stream.csv", index_label="date")
    for name, table in [("prequential_predictions", history), ("alarms", alarms), ("detector_comparison", summaries),
                        ("strategy_comparison", periods), ("alarm_error_windows", windows)]:
        table.to_csv(output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(np.arange(len(data)), original, alpha=.5, label="Original daily demand")
    axes[0].plot(np.arange(len(data)), target, label="Controlled target")
    axes[0].axvline(CHANGE, color="red", linestyle="--", label="Injected 25% reduction")
    axes[0].set(ylabel="Source demand units", title="Controlled drift on real weather and demand")
    axes[0].legend()
    for strategy, group in history.groupby("strategy", sort=False):
        axes[1].plot(group.time, group.rolling_mae_40, label=strategy)
    axes[1].axvline(CHANGE, color="red", linestyle="--")
    frozen_alarms = alarms[(alarms.strategy == "frozen") & (alarms.config == "sensitive")]
    for detector, color in [("ADWIN", "green"), ("Page-Hinkley", "orange")]:
        for j, point in enumerate(frozen_alarms[frozen_alarms.detector == detector].time):
            axes[1].axvline(point, color=color, alpha=.5, label=detector if j == 0 else None)
    axes[1].set(xlabel="Day index (0-based)", ylabel="Rolling MAE, last 40 days")
    axes[1].legend()
    save_figure(fig, output / "drift_timeline.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    periods.pivot(index="strategy", columns="period", values="mae").plot.bar(ax=axes[0], rot=0)
    axes[0].set(ylabel="MAE", title="Adaptation strategies")
    frozen = summaries[summaries.strategy == "frozen"].copy()
    labels = frozen.config + " / " + frozen.detector
    axes[1].barh(labels, frozen.delay.fillna(HORIZON + 1))
    axes[1].set(xlabel="Detection delay (101 = no timely detection)", title="Parameter sensitivity, frozen model")
    save_figure(fig, output / "strategy_comparison.png")
    window_text = markdown_table(windows) if len(windows) else "Дабылы не зарегистрированы; сравнение окон вокруг дабыла недоступно."
    online_periods = periods[periods.strategy == "online"].set_index("period")
    reset_windows = windows[(windows.strategy == "reset_adwin") & (windows.detector == "ADWIN") & (windows.config == "sensitive")]
    reset_description = "Чувствительный ADWIN не вызвал переобучение; эффект сброса в этом запуске не измерен."
    if not reset_windows.empty:
        last_reset = reset_windows.iloc[-1]
        reset_description = (
            f"После сброса в день {int(last_reset.alarm_time)} MAE изменился с {last_reset.mae_20_before:.2f} "
            f"на {last_reset.mae_20_after:.2f} (окна {int(last_reset.before_n)} и {int(last_reset.after_n)} дней); "
            f"задержка относительно введённого изменения — {int(last_reset.alarm_time)-CHANGE} дней."
        )
    report = f"""# Практическая работа №12 — вариант 1

Тема: изменение зависимости дневного электропотребления от погоды и адаптация модели.
[Задание](../../Data_Mining_Practica_12.docx).

## Данные и проверяемая гипотеза

Основа — реальные 365 дней Victoria, Australia, 2014: [elecdaily](../../data/raw/elecdaily.csv),
[источник fpp2/AEMO/BOM](https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdaily.html).
Temperature — °C, WorkDay — 0/1, Demand — исходный суточный агрегат
(документация обозначает GW; в кВт·ч здесь не переводится).
Временной порядок восстановлен по описанию: 2014-01-01–2014-12-31; пропусков нет.

**Контролируемый искусственный сценарий на реальных данных:** начиная с индекса
{CHANGE} ({data.index[CHANGE]:%Y-%m-%d}) цель уменьшена на 25% при сохранении погоды и календаря.
Так моделируется гипотеза изменения поведения после тарифа; это **не установленное событие
в Австралии**. Исходный Demand сохранён отдельно. Известная граница позволяет оценить задержку.
Сигналы до неё называются ложными только относительно введённого изменения;
они также могут отражать естественную сезонность исходного ряда.

## Онлайн оценка и стратегии

Первоначальное обучение — {WARMUP} дней. StandardScaler признаков и нормировка цели
оцениваются только на этом начальном периоде и далее фиксируются.
Признаки: температура и её квадрат, рабочий день, недельные sin/cos.
Годовые sin/cos исключены: 60-дневный warmup не позволяет надёжно оценить годовую сезонность.
SGDRegressor: squared_error, eta0=.003, constant learning rate, alpha=.001,
shuffle=False, random_state=42. Каждый из последующих {len(data)-WARMUP} дней:
сначала прогноз → ошибка → детекторы → обновление. Текущий ответ считается доступным после прогноза.

Сравнены frozen (без обновления), online (partial_fit ежедневно), reset_adwin
(фиксированная модель между сигналами и переобучение на последних 60 уже известных строках после чувствительного ADWIN).
Все стратегии стартуют одинаково; известный момент изменения не передаётся моделям и детекторам.
Детекторы получают ограниченный сигнал |ошибка|/(|ошибка|+σ_warmup), а график показывает MAE
в исходной шкале. Page–Hinkley настроен на рост средней ошибки, delta=.005, min_instances=30.

## Детекторы и параметры

{markdown_table(summaries)}

Для каждой стратегии сравниваются ADWIN delta=.01/.002 и Page–Hinkley threshold=5/10.
Первый своевременный сигнал ищется в [{CHANGE}, {CHANGE+HORIZON}], задержка — время минус {CHANGE}.
Отсутствующее значение означает отсутствие своевременного сигнала; поздние сигналы считаются отдельно.
У чувствительных настроек порог доказательств ниже, но реальный результат и ложные сигналы
нужно оценивать по таблице, а не предполагать обязательное улучшение.

![Поток, ошибки и сигналы](drift_timeline.png)

## Как ошибка меняется после сигнала

{markdown_table(periods)}

Для каждого зарегистрированного сигнала ниже сравниваются до 20 предшествующих и
до 20 последующих прогнозов; день сигнала исключён. Размеры окон указаны отдельно.

{window_text}

![Сравнение стратегий и задержки](strategy_comparison.png)

Снижение ошибки после сигнала не доказывает эффект самого детектора:
online обновляется и без сигнала; только reset_adwin применяет явный сброс.
Изменение сезонности или дисперсии тоже способно вызвать сигнал.
{reset_description}
У online MAE до изменения — {online_periods.loc['before_change', 'mae']:.2f},
в первые 40 дней после — {online_periods.loc['first_40_after', 'mae']:.2f},
в поздний период — {online_periods.loc['late_after', 'mae']:.2f}:
ежедневное обновление само по себе не обеспечило снижение ошибки в этом сезонном потоке.
У frozen ошибка сразу после вмешательства уменьшилась: снижение цели частично
компенсировало прежнее смещение модели. Поэтому рост ошибки не является необходимым
следствием каждого изменения зависимости.

## Вывод

Построен хронологический регрессионный поток дневного потребления с известным искусственным изменением.
ADWIN и Page–Hinkley проверены с двумя комбинациями параметров на ошибках до обновления модели.
Таблицы показывают задержки, ранние сигналы и изменение MAE у трёх стратегий на одном потоке.
Результат зависит от заданной силы изменения и одного года наблюдений; реальную причину тарифного сдвига детекторы не устанавливают.

Запуск: `PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab12_concept_drift`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    save_environment(output, [RAW_DIR / "elecdaily.csv"], synthetic_intervention={"change_index": CHANGE, "multiplier": .75},
                     warmup=WARMUP, detector_configs=CONFIGS, prequential=True, reset_window=60)
    print(periods.to_string(index=False))
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
