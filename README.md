# Лабораторные работы по Data Mining

Проект полностью унифицирован вокруг набора **Breast Cancer Wisconsin
Diagnostic (WDBC)**. Все лабораторные используют один загрузчик, одну схему из
30 числовых признаков и единый target: `diagnosis` (`B` — benign,
`M` — malignant).

## Структура

```text
data/
  raw/                         # единственный исходный датасет WDBC
  processed/                   # данные, подготовленные лабораторной №1
docs/assignments/              # исходные методические материалы
src/breast_cancer_labs/
  data.py                      # загрузка, валидация и очистка
  labs/                        # лабораторные №1–5, №7 и №8
results/
  lab01_preprocessing/         # таблицы, графики и report.md
  ...
run_all.py                     # последовательный запуск всех работ
tests/                         # проверки общей схемы данных
```

Лабораторной №6 в исходном проекте не было, поэтому она не добавлялась.

## Лабораторные

1. Очистка, контроль качества, IQR-анализ и стандартизация.
2. Pearson/Spearman/Kendall, корреляционная матрица и связь с диагнозом.
3. Линейная регрессия: `mean_radius` → `worst_radius`.
4. Статистическая проверка различий `mean_radius` между диагнозами.
5. K-means, Agglomerative, DBSCAN и PCA-визуализация.
7. Apriori и FP-Growth на дискретизированных клинических признаках.
8. Gradient Boosting и XGBoost для классификации диагноза.

## Запуск

В репозитории уже есть окружение `.venv` с необходимыми зависимостями:

```bash
.venv/bin/python run_all.py
```

Для нового окружения:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py
```

Отдельную работу можно выполнить так:

```bash
PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab05_clustering
```

Проверки:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Все сценарии используют фиксированный `random_state=42`, backend `Agg` для
графиков и при повторном запуске обновляют только собственный каталог в
`results/`.
