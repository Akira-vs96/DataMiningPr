"""Lab 08: Gradient Boosting and XGBoost diagnosis classification."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.data import (
    FEATURE_NAMES,
    RANDOM_STATE,
    class_distribution,
    load_clean_dataset,
    split_features_target,
)
from breast_cancer_labs.plotting import apply_style, save_figure


def build_xgb(**overrides: float | int) -> XGBClassifier:
    parameters: dict[str, float | int | str] = {
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
    }
    parameters.update(overrides)
    return XGBClassifier(**parameters)


def evaluate(name, model, X_train, X_test, y_train, y_test) -> dict[str, object]:
    train_prediction = model.predict(X_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_prediction),
        "test_accuracy": accuracy_score(y_test, prediction),
        "precision_malignant": precision_score(y_test, prediction, zero_division=0),
        "recall_malignant": recall_score(y_test, prediction, zero_division=0),
        "f1_malignant": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "overfitting_gap": accuracy_score(y_train, train_prediction) - accuracy_score(y_test, prediction),
    }


def parameter_experiment(parameter, values, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for value in values:
        model = build_xgb(**{parameter: value}).fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                parameter: value,
                "train_accuracy": model.score(X_train, y_train),
                "test_accuracy": model.score(X_test, y_test),
                "roc_auc": roc_auc_score(y_test, probability),
                "overfitting_gap": model.score(X_train, y_train) - model.score(X_test, y_test),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab08_boosting")
    data = load_clean_dataset()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.10,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": build_xgb(),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    comparison = pd.DataFrame(
        [evaluate(name, model, X_train, X_test, y_train, y_test) for name, model in models.items()]
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    distribution = class_distribution(data)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    confusion_rows = []
    for axis, (name, model) in zip(axes, models.items()):
        prediction = model.predict(X_test)
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        confusion_rows.append(
            {"model": name, "TN": matrix[0, 0], "FP": matrix[0, 1], "FN": matrix[1, 0], "TP": matrix[1, 1]}
        )
        ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"]).plot(
            ax=axis, cmap="Blues", colorbar=False
        )
        axis.set_title(name)
    save_figure(fig, output_dir / "confusion_matrices.png")
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrix_values.csv", index=False)

    fig, axis = plt.subplots(figsize=(7, 5.5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=axis)
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--")
    axis.set_title("ROC curves — malignant is the positive class")
    save_figure(fig, output_dir / "roc_curves.png")

    xgb_model = models["XGBoost"]
    importance = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(output_dir / "xgb_feature_importance.csv", index=False)
    top = importance.head(10).sort_values("importance")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["feature"], top["importance"], color="#457b9d")
    axis.set(xlabel="Importance", title="XGBoost top-10 feature importance")
    save_figure(fig, output_dir / "xgb_feature_importance_top10.png")

    experiments = {
        "n_estimators": parameter_experiment(
            "n_estimators", [50, 100, 200, 300], X_train, X_test, y_train, y_test
        ),
        "max_depth": parameter_experiment(
            "max_depth", [2, 3, 4, 5], X_train, X_test, y_train, y_test
        ),
    }
    for parameter, table in experiments.items():
        table.to_csv(output_dir / f"experiment_{parameter}.csv", index=False)
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(table[parameter], table["train_accuracy"], marker="o", label="Train")
        axis.plot(table[parameter], table["test_accuracy"], marker="o", label="Test")
        axis.set(xlabel=parameter, ylabel="Accuracy", title=f"XGBoost: {parameter} effect")
        axis.set_xticks(table[parameter])
        axis.legend()
        save_figure(fig, output_dir / f"experiment_{parameter}.png")

    best = comparison.loc[comparison["roc_auc"].idxmax()]
    top_features = ", ".join(importance.head(5)["feature"])
    report = f"""# Лабораторная работа №8 — Gradient Boosting и XGBoost

В классификации участвуют **{len(data)}** объектов и **{len(FEATURE_NAMES)}**
признаков. Использовано стратифицированное разбиение 70/30 с
random_state={RANDOM_STATE}; положительный класс — malignant.

По ROC-AUC лучший результат показал **{best['model']}**:
ROC-AUC={best['roc_auc']:.4f}, accuracy={best['test_accuracy']:.4f},
recall malignant={best['recall_malignant']:.4f}. В медицинском контексте FN
(пропущенный malignant) особенно важен, поэтому accuracy рассматривается только
вместе с recall, F1, ROC-AUC и confusion matrix.

Пять наиболее используемых XGBoost признаков: {top_features}. Feature importance
показывает использование признака моделью, но не доказывает причинность.
Эксперименты отдельно оценивают `n_estimators` и `max_depth`, включая разрыв
между train и test accuracy как индикатор переобучения.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(comparison.round(4).to_string(index=False))
    print(f"Top XGBoost features: {top_features}")


if __name__ == "__main__":
    main()
