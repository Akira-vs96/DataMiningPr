"""Practical 11, variant 1: explain the most confident Wine prediction."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.plotting import apply_style, save_figure
from breast_cancer_labs.practical_common import markdown_table, save_environment


def main() -> None:
    apply_style()
    output = prepare_output_dir("lab11_explainable_ai")
    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Wine contains missing values")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1).fit(X_train, y_train)
    prediction, probability = model.predict(X_test), model.predict_proba(X_test)
    scores = pd.DataFrame([{"accuracy": accuracy_score(y_test, prediction), "macro_f1": f1_score(y_test, prediction, average="macro")}])
    permutation = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, scoring="accuracy")
    explanation = shap.TreeExplainer(model)(X_test)
    if explanation.values.shape != (len(X_test), X.shape[1], len(model.classes_)):
        raise ValueError(f"Unsupported SHAP shape: {explanation.values.shape}")
    # Special requirement: choose maximum predicted confidence over the entire test.
    row = int(np.argmax(probability.max(axis=1)))
    class_position = int(probability[row].argmax())
    selected_class = int(model.classes_[class_position])
    class_exp = explanation[:, :, class_position]
    reconstruction = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(reconstruction, probability, atol=1e-6)
    importance = pd.DataFrame({"feature": X.columns, "rf_impurity": model.feature_importances_,
                               "permutation_mean": permutation.importances_mean,
                               "permutation_std": permutation.importances_std,
                               "mean_absolute_shap_all_classes": np.abs(explanation.values).mean(axis=(0, 2)),
                               "mean_absolute_shap_selected_class": np.abs(class_exp.values).mean(axis=0)})
    lime = LimeTabularExplainer(X_train.to_numpy(), feature_names=list(X.columns),
                               class_names=list(wine.target_names), mode="classification",
                               discretize_continuous=True, random_state=42)
    def predict_proba(values):
        return model.predict_proba(pd.DataFrame(values, columns=X.columns))
    local = lime.explain_instance(X_test.iloc[row].to_numpy(), predict_proba,
                                  labels=(class_position,), num_features=8, num_samples=2000)
    lime_table = pd.DataFrame(local.as_list(label=class_position), columns=["condition", "weight"])
    shap_table = pd.DataFrame({"feature": X.columns, "value": X_test.iloc[row].to_numpy(),
                               "shap_probability_contribution": class_exp.values[row]}).sort_values("shap_probability_contribution", key=abs, ascending=False)
    selected = {"source_row": int(X_test.index[row]), "test_position": row,
                "actual_class": int(y_test.iloc[row]), "predicted_class": selected_class,
                "class_name": str(wine.target_names[selected_class]),
                "probabilities": probability[row].tolist(),
                "maximum_test_confidence": float(probability.max()),
                "confidence_ties": int(np.isclose(probability.max(axis=1), probability.max()).sum()),
                "shap_base_probability": float(class_exp.base_values[row]),
                "shap_sum": float(class_exp.values[row].sum()),
                "lime_local_r2": float(local.score), "lime_intercept": float(local.intercept[class_position]),
                "lime_local_prediction": float(np.asarray(local.local_pred).ravel()[0])}
    (output / "selected_observation.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    scores.to_csv(output / "metrics.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    lime_table.to_csv(output / "lime_local.csv", index=False)
    shap_table.to_csv(output / "shap_local.csv", index=False)
    pd.DataFrame(classification_report(y_test, prediction, output_dict=True, zero_division=0)).T.to_csv(output / "classification_report.csv")
    pd.DataFrame({"actual": y_test, "predicted": prediction, **{f"p_class_{i}": probability[:, i] for i in range(3)}}).to_csv(output / "test_predictions.csv", index_label="source_row")
    pd.DataFrame({"target": y, "split": np.where(X.index.isin(X_test.index), "test", "train")}).to_csv(output / "data_split.csv", index_label="source_row")
    pd.DataFrame(class_exp.values, index=X_test.index, columns=X.columns).to_csv(output / "shap_test_selected_class.csv", index_label="source_row")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, column in zip(axes, ["rf_impurity", "permutation_mean", "mean_absolute_shap_all_classes"]):
        importance.set_index("feature")[column].nlargest(8).sort_values().plot.barh(ax=ax)
        ax.set_title(column)
    save_figure(fig, output / "global_importance.png")
    shap.plots.beeswarm(class_exp, max_display=10, show=False)
    save_figure(plt.gcf(), output / "shap_beeswarm.png")
    shap.plots.waterfall(class_exp[row], max_display=10, show=False)
    save_figure(plt.gcf(), output / "shap_waterfall.png")
    save_figure(local.as_pyplot_figure(label=class_position), output / "lime_local.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(wine.target_names, probability[row])
    ax.set(ylim=(0, 1.05), ylabel="Probability", title=f"Most confident test row: {selected['source_row']}")
    save_figure(fig, output / "selected_probabilities.png")
    local.save_to_file(str(output / "lime_explanation.html"))
    report = f"""# Практическая работа №11 — вариант 1

Тема: объяснение химических признаков классов Wine через SHAP и LIME.
[Задание](../../Data_Mining_Practica_11.docx), [источник Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset).

## Данные и качество модели

Использован встроенный sklearn Wine: {len(X)} образцов, {X.shape[1]} химических признаков,
три класса с численностями {y.value_counts().sort_index().to_dict()}.
Пропуски: {int(X.isna().sum().sum())}; точные повторы признаков: {int(X.duplicated().sum())}.
Стратифицированное разбиение 75/25: train={len(X_train)}, test={len(X_test)}.
Random Forest: 120 деревьев, random_state=42. Обучение использует только train.

{markdown_table(scores)}

## Общая важность

{markdown_table(importance.sort_values('mean_absolute_shap_all_classes', ascending=False).head(8))}

Встроенная impurity-важность, падение test-Accuracy при перестановке (8 повторов)
и среднее |SHAP| по всем test-объектам и классам измеряют разные свойства.
Коррелирующие химические признаки могут делить важность; нулевое падение Accuracy
не означает, что признак никогда не используется деревьями.

![Общие важности](global_importance.png)
![SHAP beeswarm выбранного класса](shap_beeswarm.png)

## Специальное задание: самый уверенный прогноз

По максимуму вероятности **среди всех test-объектов** выбрана исходная строка
{selected['source_row']}. Максимум — {probability[row, class_position]:.4f};
таких максимумов {selected['confidence_ties']}, при равенстве берётся первый в порядке test.
Истинный класс — {selected['actual_class']}, прогноз — {selected_class} ({selected['class_name']}).
Вероятности классов 0/1/2: {probability[row].round(4).tolist()}.
Выбор по уверенности не является отдельной оценкой качества модели и не гарантирует правильность.

![Вероятности](selected_probabilities.png)

SHAP объясняет **вероятность класса {selected_class}**, а не логит.
База {selected['shap_base_probability']:.6f} + сумма вкладов {selected['shap_sum']:.6f}
= {probability[row, class_position]:.6f}; аддитивность проверена для всех test-объектов и классов с допуском 1e-6.
TreeExplainer использует распределение путей деревьев обученного леса.

{markdown_table(shap_table.head(8))}

![SHAP waterfall](shap_waterfall.png)

LIME использует только train для построения окрестности и границ дискретизации,
2000 возмущённых объектов, 8 условий. Положительные веса поддерживают класс {selected_class},
отрицательные уменьшают его локальный прогноз.
Взвешенный R² локального приближения — {local.score:.4f}, прогноз приближения — {selected['lime_local_prediction']:.4f}.

{markdown_table(lime_table)}

![LIME](lime_local.png)

## Два сходства и два различия

1. Оба метода объясняют выход уже обученного Random Forest для одного и того же объекта и класса.
2. Оба показывают положительное и отрицательное влияние признаков относительно своего контекста объяснения.
3. TreeSHAP раскладывает вероятность относительно базового значения аддитивно;
   LIME обучает взвешенное локальное линейное приближение, которое может не воспроизвести вероятность точно.
4. SHAP в данном опыте использует структуру деревьев; LIME зависит от случайных возмущений,
   дискретизации и ширины окрестности. Поэтому порядок и знаки важностей могут различаться.

## Вывод

Качество Random Forest проверено на отложенной выборке, затем объяснён самый уверенный test-прогноз.
Общие ранги и локальные вклады помогают понять использование химических признаков моделью.
SHAP и LIME отвечают на близкие вопросы разными способами и не обязаны совпадать.
Эти объяснения описывают модельные зависимости, но не доказывают причинное влияние состава на класс вина.

Запуск: `PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab11_explainable_ai`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    save_environment(output, [], dataset="sklearn.load_wine", split="stratified 75/25", selected_observation=selected)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
