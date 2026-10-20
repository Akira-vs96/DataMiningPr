"""Lab 03: simple linear regression for two continuous WDBC measurements."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.data import RANDOM_STATE, load_clean_dataset
from breast_cancer_labs.plotting import apply_style, save_figure


PREDICTOR = "mean_radius"
TARGET = "worst_radius"


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab03_regression")
    data = load_clean_dataset()
    X = data[[PREDICTOR]]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = LinearRegression().fit(X_train, y_train)
    prediction = model.predict(X_test)
    residuals = y_test.to_numpy() - prediction
    r2 = r2_score(y_test, prediction)
    n, predictors = len(y_test), X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - predictors - 1)
    metrics = pd.DataFrame(
        {
            "metric": ["R2", "Adjusted R2", "MAE", "MSE", "RMSE"],
            "value": [
                r2,
                adjusted_r2,
                mean_absolute_error(y_test, prediction),
                mean_squared_error(y_test, prediction),
                np.sqrt(mean_squared_error(y_test, prediction)),
            ],
        }
    )
    metrics.to_csv(output_dir / "regression_metrics.csv", index=False)

    coefficients = pd.DataFrame(
        {"term": ["intercept", PREDICTOR], "value": [model.intercept_, model.coef_[0]]}
    )
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)
    predictions = pd.DataFrame(
        {
            "row_index": y_test.index,
            "actual": y_test.to_numpy(),
            "predicted": prediction,
            "residual": residuals,
        }
    ).sort_values("row_index")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    grid = pd.DataFrame({PREDICTOR: np.linspace(X[PREDICTOR].min(), X[PREDICTOR].max(), 200)})
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(X_train[PREDICTOR], y_train, s=20, alpha=0.45, label="Train")
    axis.scatter(X_test[PREDICTOR], y_test, s=32, alpha=0.75, label="Test")
    axis.plot(grid[PREDICTOR], model.predict(grid), color="#d62828", linewidth=2, label="Regression")
    axis.set(xlabel=PREDICTOR, ylabel=TARGET, title="Simple linear regression")
    axis.legend()
    save_figure(fig, output_dir / "regression_fit.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(prediction, residuals, color="#f4a261", edgecolor="white")
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel=f"Predicted {TARGET}", ylabel="Residual", title="Residual diagnostics")
    save_figure(fig, output_dir / "residual_plot.png")

    report = f"""# Лабораторная работа №3 — линейная регрессия

Модель прогнозирует `{TARGET}` по `{PREDICTOR}` на независимой тестовой выборке
20% (random_state={RANDOM_STATE}).

Уравнение: **{TARGET} = {model.intercept_:.4f} + {model.coef_[0]:.4f} × {PREDICTOR}**.

- R² = **{r2:.4f}**;
- adjusted R² = **{adjusted_r2:.4f}**;
- RMSE = **{metrics.loc[metrics['metric'] == 'RMSE', 'value'].iloc[0]:.4f}**;
- VIF единственного предиктора по определению равен 1.

Коэффициент описывает среднее изменение worst radius при увеличении mean radius
на единицу. Это ассоциативная прогнозная модель, а не причинное заключение.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(coefficients.round(5).to_string(index=False))
    print(metrics.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
