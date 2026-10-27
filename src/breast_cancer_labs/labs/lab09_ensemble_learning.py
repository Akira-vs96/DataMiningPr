"""Lab 09, variant 1: Penguins classification with Stacking and Blending."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from breast_cancer_labs.config import PENGUINS_ARCHIVE, PROCESSED_DIR, prepare_output_dir
from breast_cancer_labs.data import RANDOM_STATE
from breast_cancer_labs.penguins_data import (
    CATEGORICAL_FEATURES,
    DATA_MEMBER,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    SPECIES,
    load_raw_penguins_dataset,
    remove_incomplete_rows,
)
from breast_cancer_labs.plotting import apply_style, save_figure


MODEL_NAMES = {"dt": "Decision Tree", "rf": "Random Forest", "gb": "Gradient Boosting"}


def build_preprocessor() -> ColumnTransformer:
    """Encode categories on each training subset; numeric values pass unchanged."""
    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_base_models() -> list[tuple[str, Pipeline]]:
    """Intentionally weak models for the requested underfitting experiment."""
    classifiers = [
        ("dt", DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=50,
                max_depth=1,
                max_features=1,
                random_state=RANDOM_STATE,
                n_jobs=5,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=10,
                learning_rate=0.02,
                max_depth=1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    return [
        (name, Pipeline([("preprocessing", build_preprocessor()), ("classifier", model)]))
        for name, model in classifiers
    ]


def build_meta_model() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    )


def build_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_stacking(base_models: list[tuple[str, Pipeline]]) -> StackingClassifier:
    # Preprocessing belongs INSIDE each estimator, so it is refit in every OOF fold.
    return StackingClassifier(
        estimators=base_models,
        final_estimator=build_meta_model(),
        cv=build_folds(),
        stack_method="predict_proba",
        passthrough=False,
    )


def ordered_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Keep probability columns aligned to the same explicit species order."""
    positions = [list(model.classes_).index(species) for species in SPECIES]
    return model.predict_proba(X)[:, positions]


def probability_features(models: list[tuple[str, Pipeline]], X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.hstack([ordered_probabilities(model, X) for _, model in models]),
        index=X.index,
        columns=[f"{name}__p_{species}" for name, _ in models for species in SPECIES],
    )


@dataclass
class FittedBlending:
    base_models: list[tuple[str, Pipeline]]
    meta_model: Pipeline
    validation_features: pd.DataFrame

    @property
    def classes_(self) -> np.ndarray:
        return self.meta_model.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_model.predict(probability_features(self.base_models, X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_model.predict_proba(probability_features(self.base_models, X))


def fit_blending(
    base_models: list[tuple[str, Pipeline]],
    X_base: pd.DataFrame,
    y_base: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> FittedBlending:
    """Train base pipelines on base-train and the meta-model on hold-out only."""
    if not X_base.index.intersection(X_valid.index).empty:
        raise ValueError("Blending base-train and validation rows must be disjoint")
    fitted = [(name, clone(model).fit(X_base, y_base)) for name, model in base_models]
    Z_valid = probability_features(fitted, X_valid)
    meta = build_meta_model().fit(Z_valid, y_valid)
    # Keep the fitted base models: refitting on validation would change the meta inputs.
    return FittedBlending(fitted, meta, Z_valid)


def evaluate(name: str, y_true: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, prediction),
        "precision_macro": precision_score(y_true, prediction, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, prediction, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, prediction, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(
            y_true, probability, labels=SPECIES, multi_class="ovr", average="macro"
        ),
    }


def markdown_table(table: pd.DataFrame) -> str:
    """Render reports without an optional tabulate dependency."""
    def format_cell(value) -> str:
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |",
            *["| " + " | ".join(map(format_cell, row)) + " |" for row in table.itertuples(index=False, name=None)],
        ]
    )


def write_report(output_dir, raw, data, scores, distribution, missing, splits, matrices) -> None:
    parameter_rows = []
    for name, pipeline in build_base_models():
        parameters = pipeline.named_steps["classifier"].get_params(deep=False)
        settings = ", ".join(
            f"{key}={parameters[key]}"
            for key in ["max_depth", "n_estimators", "learning_rate", "max_features", "random_state"]
            if key in parameters
        )
        parameter_rows.append({"model": MODEL_NAMES[name], "parameters": settings})
    parameter_table = markdown_table(pd.DataFrame(parameter_rows))
    accuracy_winners = scores.loc[np.isclose(scores.accuracy, scores.accuracy.max()), "model"]
    f1_winners = scores.loc[np.isclose(scores.f1_macro, scores.f1_macro.max()), "model"]
    indexed = scores.set_index("model")
    base_best = indexed.loc[list(MODEL_NAMES.values()), "f1_macro"].max()
    stack = indexed.loc["Stacking"]
    blend = indexed.loc["Blending"]
    stacking_errors = int(matrices["Stacking"].sum() - np.trace(matrices["Stacking"]))
    blending_errors = int(matrices["Blending"].sum() - np.trace(matrices["Blending"]))
    off_diagonal = matrices["Stacking"].copy()
    np.fill_diagonal(off_diagonal, 0)
    actual, predicted = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
    confusion_description = (
        f"самая частая ошибка Stacking — {SPECIES[actual]} → {SPECIES[predicted]} "
        f"(число объектов: {off_diagonal[actual, predicted]})"
        if off_diagonal.any() else "Stacking правильно определил все виды"
    )
    stack_comparison = (
        "Stacking повторил лучший macro-F1 отдельных моделей, поэтому дополнительного выигрыша от объединения моделей на этом test нет"
        if np.isclose(stack.f1_macro, base_best)
        else f"Macro-F1 Stacking относительно лучшей отдельной модели отличается на {(stack.f1_macro - base_best) * 100:+.2f} процентного пункта"
    )
    counts = splits.set_index("split")["rows"]
    train_size, test_size = int(counts["train"]), int(counts["test"])
    base_size, valid_size = int(counts["blending_base"]), int(counts["blending_validation"])
    report = f"""# Практическая работа №9 — вариант 1

**Тема:** Ensemble Learning: Stacking и Blending.  
**Задача:** классификация видов пингвинов (Penguins species classification).

## Цель

Сравнить Decision Tree, Random Forest, Gradient Boosting, Stacking и Blending
на одной тестовой выборке, обработать числовые и категориальные признаки
без утечки данных и оценить ошибки по видам.
В текущем учебном эксперименте базовые модели намеренно упрощены,
чтобы продемонстрировать недообучение и объединение их прогнозов метамоделью.

## Данные

Источник эксперимента — предоставленный пользователем архив
[`archive.zip`](../../data/raw/archive.zip), файл `{DATA_MEMBER}`.
В методичке указан Palmer Penguins; в этой работе используется
предоставленная таблица: **{len(raw)} исходных строк, {len(FEATURE_NAMES)} признаков и 3 класса**.
После удаления строк с пропусками осталось **{len(data)} полных наблюдений**.
В архиве нет описания происхождения и способа формирования этих {len(raw)} строк;
его результаты относятся к этой версии данных.
Исходное задание: [практическая №9](../../docs/assignments/practical_09.docx).

Целевая переменная — `species`: Adelie, Chinstrap, Gentoo.
Числовые признаки: `culmen_length_mm` (длина клюва), `culmen_depth_mm`
(глубина клюва), `flipper_length_mm` (длина ласта), `body_mass_g` (масса).
Категориальные признаки: `island` (остров) и `sex` (пол).
Точных дубликатов в очищенной таблице: **{int(data.duplicated().sum())}**.
Удалены **{len(raw) - len(data)} строк** хотя бы с одним пропуском;
оставшиеся строки содержат **{int(data.isna().sum().sum())} пропусков**.
Полные наблюдения сохранены в
[`penguins_clean.csv`](../../data/processed/penguins_clean.csv),
исключённые исходные строки и причины удаления — в `removed_rows.csv`.
`species` и номер строки не включаются в признаки.

### Баланс классов

{markdown_table(distribution)}

Доли классов различаются, поэтому вместе с Accuracy рассчитываются
macro-F1, macro-Precision, macro-Recall и multiclass ROC-AUC (OvR, macro).
Macro-усреднение придаёт одинаковый вес каждому виду.

### Пропуски до и после удаления строк

{markdown_table(missing)}

Применяется удаление неполных наблюдений: `raw.dropna()` до разбиения на выборки.
Значения не заполняются; `SimpleImputer` исключён из всех моделей.
Числовые признаки передаются без изменений, категориальные кодируются через
One-Hot Encoding (`handle_unknown='ignore'`).
`ColumnTransformer` находится внутри каждого базового `Pipeline`.
Для отдельных моделей кодировщик обучается на train, для Stacking — заново
внутри обучающей части каждого OOF-фолда, для Blending — на base-train.
Метамодель использует `StandardScaler` и `LogisticRegression(max_iter=2000)`;
масштабирование метапризнаков обучается вместе с метамоделью.

## Разбиение и обучение

`random_state={RANDOM_STATE}` во всех случайных операциях; разбиения стратифицированы по `species`.
Параметры намеренно ограничены в учебном эксперименте;
подбор на максимум качества не выполняется.

{markdown_table(splits)}

- Отдельные модели обучены на {train_size} строках train.
- **Stacking:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`;
  каждая обучающая строка получает вероятности от моделей, которые её не видели.
  Метамодель обучена на OOF-матрице размера {train_size} × 9
  (3 модели × 3 вида); затем базовые модели обучены на всём train для инференса.
- **Blending:** базовые модели обучены на {base_size} строках, метамодель —
  на вероятностях для {valid_size} отдельных validation-строк (матрица {valid_size} × 9).
  После этого базовые модели сохраняются без переобучения.
- **Test:** общие {test_size} строк для итоговой оценки всех пяти моделей.

Распределение каждой сохранённой строки записано в `data_split.csv`:
`source_row` — индекс строки данных с нуля, `oof_fold` — номер фолда,
в котором строка служит OOF-validation, `blending_role` — base или validation.
Test не участвует в обучении базовых моделей, кодировщиков,
масштабирования или метамоделей.

## Код моделей и воспроизведение

Полная реализация предобработки, моделей, метрик и графиков:
[`lab09_ensemble_learning.py`](../../src/breast_cancer_labs/labs/lab09_ensemble_learning.py).
Загрузчик: [`penguins_data.py`](../../src/breast_cancer_labs/penguins_data.py).
Актуальные настройки автоматически считываются из базовых моделей:

{parameter_table}

Глубина деревьев ограничена одним уровнем, у Random Forest оставлено одно
дерево, у Gradient Boosting — один этап с `learning_rate=0.01`.
Метамодели используют вероятности, поэтому даже при слабых итоговых ответах
отдельных моделей различия их вероятностей могут помогать классификации.
Остальные параметры используются по умолчанию из версии scikit-learn,
зафиксированной в `environment.json`.

Ключевая последовательность (запускается из корня с `PYTHONPATH=src`):

```python
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from breast_cancer_labs.labs.lab09_ensemble_learning import (
    build_base_models, build_stacking, fit_blending,
    evaluate, ordered_probabilities,
)
from breast_cancer_labs.penguins_data import FEATURE_NAMES, load_penguins_dataset

data = load_penguins_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    data[FEATURE_NAMES], data['species'], test_size=0.30,
    random_state=42, stratify=data['species'],
)
base_models = build_base_models()  # каждый содержит свой preprocessing Pipeline
models = {{name: clone(model).fit(X_train, y_train) for name, model in base_models}}
models['Stacking'] = build_stacking(base_models).fit(X_train, y_train)
X_base, X_valid, y_base, y_valid = train_test_split(
    X_train, y_train, test_size=0.30, random_state=42, stratify=y_train,
)
models['Blending'] = fit_blending(base_models, X_base, y_base, X_valid, y_valid)
for name, model in models.items():
    print(evaluate(name, y_test, model.predict(X_test), ordered_probabilities(model, X_test)))
```

Полный запуск с сохранением всех результатов:

```bash
PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab09_ensemble_learning
```

## Метрики на test

{markdown_table(scores)}

### График 1 — Accuracy и macro-F1

![Сравнение метрик](model_comparison.png)

Столбцы показывают результаты на одинаковых {test_size} тестовых объектах;
ось Y начинается с нуля. При отличающихся долях видов macro-F1 помогает
оценивать качество с равным весом каждого класса.

### График 2 — матрицы ошибок по видам

![Матрицы ошибок пяти моделей](confusion_matrices.png)

Строки — истинные виды, столбцы — предсказанные.
Диагональ соответствует правильным ответам, остальные клетки показывают
направление и число ошибок; для всех моделей используется одна цветовая шкала.
Подробные precision/recall/F1 по видам находятся в `classification_report.csv`,
численные матрицы — в `confusion_matrix_values.csv`.

## Сравнительный анализ (7 предложений)

1. Максимальная Accuracy равна {scores.accuracy.max():.4f}; её показали {', '.join(accuracy_winners)}.
2. Максимальный macro-F1 равен {scores.f1_macro.max():.4f}; лучший результат у {', '.join(f1_winners)}.
3. {stack_comparison}.
4. ROC-AUC Stacking равен {stack.roc_auc_ovr_macro:.4f}, а Blending — {blend.roc_auc_ovr_macro:.4f}; эта метрика оценивает ранжирование вероятностей и может различаться даже при одинаковых итоговых классах.
5. В Stacking метамодель использует OOF-прогнозы для всех {train_size} train-строк, а в Blending — только {valid_size} validation-строк при обучении базовых моделей на {base_size} строках.
6. Число ошибок Stacking — {stacking_errors}, Blending — {blending_errors} на {test_size} тестовых объектах; {confusion_description}.
7. Строки с пропусками удалены до разбиения, а кодировщики обучаются внутри соответствующих train-частей, поэтому test и OOF-validation не передают категории в обучение базовых моделей.

## Вывод (4 предложения)

Выполнена многоклассовая классификация видов пингвинов на {len(data)} полных наблюдениях после удаления {len(raw) - len(data)} строк с пропусками.
Сравнение всех пяти моделей проведено на одном стратифицированном тестовом разбиении с Accuracy, macro-F1 и дополнительными метриками.
Stacking использует OOF-вероятности, а Blending — отдельную validation-часть, что меняет объём данных для обучения метамодели.
Результаты характеризуют один фиксированный split полных наблюдений; удаление строк изменяет состав test, поэтому сравнение с прежним заполнением пропусков не является оценкой на одинаковой тестовой выборке.

## Сохранённые материалы

- `model_comparison.csv`, `classification_report.csv` — общие и поклассовые метрики.
- `test_predictions.csv`, `test_probabilities.csv` — ответы и вероятности для проверки метрик.
- `data_split.csv`, `split_summary.csv` — роли строк и состав выборок.
- `blending_validation_features.csv` — validation-метапризнаки Blending и целевой класс.
- `missing_values.csv`, `class_distribution.csv`, `descriptive_statistics.csv` — описание данных.
- `removed_rows.csv` — исключённые исходные строки с перечнем отсутствующих полей.
- `environment.json` — версии библиотек, параметры разбиения и SHA-256 входного архива.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    apply_style()
    raw = load_raw_penguins_dataset()
    data = remove_incomplete_rows(raw)
    output_dir = prepare_output_dir("lab09_ensemble_learning")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DIR / "penguins_clean.csv")
    removed = raw.loc[~raw.index.isin(data.index)].copy()
    removed["missing_columns"] = removed.isna().apply(
        lambda row: ", ".join(row.index[row]), axis=1
    )
    removed.to_csv(output_dir / "removed_rows.csv")
    X, y = data[FEATURE_NAMES], data["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y,
    )
    X_base, X_valid, y_base, y_valid = train_test_split(
        X_train, y_train, test_size=0.30, random_state=RANDOM_STATE, stratify=y_train,
    )
    base_models = build_base_models()
    models = {
        MODEL_NAMES[name]: clone(model).fit(X_train, y_train)
        for name, model in base_models
    }
    models["Stacking"] = build_stacking(base_models).fit(X_train, y_train)
    blending = fit_blending(base_models, X_base, y_base, X_valid, y_valid)
    models["Blending"] = blending

    scores, report_rows, confusion_rows, probability_rows = [], [], [], []
    predictions = pd.DataFrame({"actual_species": y_test}).sort_index()
    matrices = {}
    for name, model in models.items():
        prediction = model.predict(X_test)
        probability = ordered_probabilities(model, X_test)
        scores.append(evaluate(name, y_test, prediction, probability))
        predictions[name] = pd.Series(prediction, index=X_test.index)
        details = classification_report(
            y_test, prediction, labels=SPECIES, output_dict=True, zero_division=0,
        )
        for label in [*SPECIES, "macro avg", "weighted avg"]:
            report_rows.append({"model": name, "species_or_average": label, **details[label]})
        matrix = confusion_matrix(y_test, prediction, labels=SPECIES)
        matrices[name] = matrix
        for i, actual in enumerate(SPECIES):
            for j, predicted in enumerate(SPECIES):
                confusion_rows.append(
                    {"model": name, "actual_species": actual, "predicted_species": predicted, "count": int(matrix[i, j])}
                )
        probs = pd.DataFrame(probability, index=X_test.index, columns=[f"p_{c}" for c in SPECIES])
        probs.insert(0, "model", name)
        probs.insert(1, "actual_species", y_test)
        probability_rows.append(probs)

    scores = pd.DataFrame(scores)
    scores.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv")
    pd.concat(probability_rows).sort_index().to_csv(output_dir / "test_probabilities.csv")
    pd.DataFrame(report_rows).to_csv(output_dir / "classification_report.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrix_values.csv", index=False)
    blending.validation_features.assign(species=y_valid).sort_index().to_csv(
        output_dir / "blending_validation_features.csv"
    )

    counts = y.value_counts().reindex(SPECIES)
    distribution = pd.DataFrame({"species": SPECIES, "count": counts.to_numpy(), "share": (counts / len(y)).to_numpy()})
    missing = raw.isna().sum().rename_axis("column").reset_index(name="missing_count_before")
    missing["missing_share_before"] = missing["missing_count_before"] / len(raw)
    missing["missing_count_after"] = missing["column"].map(data.isna().sum())
    missing["missing_share_after"] = missing["missing_count_after"] / len(data)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    missing.to_csv(output_dir / "missing_values.csv", index=False)
    data[NUMERIC_FEATURES].describe().to_csv(output_dir / "descriptive_statistics.csv")

    assignments = pd.DataFrame({"species": y, "split": "train", "blending_role": "unused_test"})
    assignments.loc[X_test.index, "split"] = "test"
    assignments.loc[X_base.index, "blending_role"] = "base"
    assignments.loc[X_valid.index, "blending_role"] = "validation"
    assignments["oof_fold"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    for fold, (_, valid_positions) in enumerate(build_folds().split(X_train, y_train), start=1):
        assignments.loc[X_train.index[valid_positions], "oof_fold"] = fold
    assignments.to_csv(output_dir / "data_split.csv")
    splits = pd.DataFrame(
        [
            {"split": name, "rows": len(labels), **labels.value_counts().reindex(SPECIES, fill_value=0).to_dict()}
            for name, labels in [("train", y_train), ("test", y_test), ("blending_base", y_base), ("blending_validation", y_valid)]
        ]
    )
    splits.to_csv(output_dir / "split_summary.csv", index=False)
    environment = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "dataset_archive": "data/raw/archive.zip", "dataset_member": DATA_MEMBER,
        "dataset_sha256": hashlib.sha256(PENGUINS_ARCHIVE.read_bytes()).hexdigest(),
        "random_state": RANDOM_STATE, "test_size": 0.30,
        "blending_validation_fraction_of_train": 0.30, "stacking_folds": 5,
        "missing_values_policy": "drop_rows_before_split",
        "rows_before_cleaning": len(raw), "rows_removed": len(removed), "rows_after_cleaning": len(data),
        "processed_dataset": "data/processed/penguins_clean.csv",
        "experiment": "intentionally_weak_base_models",
        "base_model_parameters": {
            MODEL_NAMES[name]: pipeline.named_steps["classifier"].get_params(deep=False)
            for name, pipeline in base_models
        },
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(scores))
    for offset, metric, label, color in [(-0.19, "accuracy", "Accuracy", "#457b9d"), (0.19, "f1_macro", "Macro-F1", "#2a9d8f")]:
        bars = axis.bar(positions + offset, scores[metric], width=0.38, label=label, color=color)
        axis.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    axis.set(xticks=positions, xticklabels=scores.model, ylim=(0, 1.10), ylabel="Test metric value", title="Penguins: base models, Stacking and Blending")
    axis.tick_params(axis="x", labelrotation=15)
    axis.legend(loc="lower right")
    save_figure(fig, output_dir / "model_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    max_count = max(int(matrix.max()) for matrix in matrices.values())
    for axis, (name, matrix) in zip(axes.flat, matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=SPECIES).plot(
            ax=axis, cmap="Blues", colorbar=False, im_kw={"vmin": 0, "vmax": max_count},
        )
        axis.grid(False)
        axis.set_title(name)
    axes.flat[-1].axis("off")
    save_figure(fig, output_dir / "confusion_matrices.png")

    write_report(output_dir, raw, data, scores, distribution, missing, splits, matrices)
    print(f"Rows: {len(raw)} source, {len(removed)} removed, {len(data)} complete")
    print(scores.round(4).to_string(index=False))
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
