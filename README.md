# Лабораторные работы по Data Mining

Лабораторные №1–5, №7 и №8 используют **Breast Cancer Wisconsin
Diagnostic (WDBC)**: общий загрузчик, 30 числовых признаков и target
`diagnosis` (`B` — benign, `M` — malignant).
Практическая №9, вариант 1, использует предоставленный **Penguins** из
`archive.zip`: 1000 исходных строк, 4 числовых и 2 категориальных признака,
target `species` (Adelie, Chinstrap, Gentoo).
Строки с пропусками удаляются: остаётся 951 полное наблюдение;
очищенные данные сохраняются в `data/processed/penguins_clean.csv`.

## Структура

```text
data/
  raw/                         # исходные ZIP: WDBC и archive.zip (Penguins)
  processed/                   # подготовленные WDBC и очищенный Penguins
docs/assignments/              # исходные методические материалы
src/breast_cancer_labs/
  data.py                      # загрузка, валидация и очистка WDBC
  penguins_data.py              # загрузка Penguins и удаление строк с пропусками
  config.py                    # пути к данным и каталогам результатов
  plotting.py                  # общие настройки графиков
  labs/                        # лабораторные №1–5, №7–15 с функцией main()
results/
  lab01_preprocessing/         # таблицы, графики и report.md
  ...
run_all.py                     # последовательный запуск всех работ
tests/                         # проверки данных и защиты от утечки в №9
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
9. Вариант 1: классификация видов Penguins; Decision Tree, Random Forest,
   Gradient Boosting, Stacking (5-fold OOF) и Blending (hold-out),
   метамодель Logistic Regression.
10. Вариант 1: месячное электропотребление США (EIA), SARIMA и сезонный baseline,
    ADF, ACF/PACF, временная валидация и прогноз на 24 месяца.
11. Вариант 1: Wine, Random Forest, общая важность, SHAP и LIME для самого
    уверенного прогноза тестовой выборки.
12. Вариант 1: дневное электропотребление и температура; контролируемое искусственное
    снижение потребления на реальных данных, ADWIN/Page–Hinkley и онлайн регрессия.
13. Вариант 1: Karate Club, четыре центральности, сравнение топ-5 и Louvain.
14. Вариант 1: реальные научные аннотации OpenAlex, LDA/NMF с 3 и 5 темами.
15. Вариант 1: суточное электропотребление, Ridge и conformal-интервалы 80/90/95%,
    хронологические train/calibration/test и оценка покрытия по сезонам.

Другие варианты практик 10–15 не реализованы. Исходные задания находятся
в файлах `Data_Mining_Practica_10.docx`–`Data_Mining_Practica_15.docx` в корне.
Источники новых данных и единицы измерения описаны в [data/raw/README.md](data/raw/README.md).

## Архитектура

Каждая работа — отдельный модуль в `src/breast_cancer_labs/labs/` с точкой
входа `main()`. `run_all.py` последовательно вызывает работы, записывает
`results/run_summary.csv` и обновляет оглавление результатов. Общие пути
хранятся в `config.py`, оформление графиков — в `plotting.py`.
Каждый модуль сохраняет CSV, PNG и русскоязычный `report.md` в свой каталог.

Загрузчики WDBC и Penguins разделены, поскольку задачи имеют разные схемы
и целевые переменные. Для №9 строки с пропусками удаляются до разбиения;
категориальное кодирование встроено в каждую базовую модель через
`Pipeline` и `ColumnTransformer`:
это позволяет обучать предобработку отдельно в каждом OOF-фолде Stacking
и только на base-train в Blending. Существующие работы используют WDBC.

## Запуск

Только практики 10–15, вариант 1 (готовые данные уже сохранены, интернет не нужен):

```bash
.venv/bin/python run_all.py --labs 10 11 12 13 14 15
```

Отчёты: [№10](results/lab10_time_series/report.md),
[№11](results/lab11_explainable_ai/report.md),
[№12](results/lab12_concept_drift/report.md),
[№13](results/lab13_graph_analysis/report.md),
[№14](results/lab14_topic_modeling/report.md),
[№15](results/lab15_prediction_intervals/report.md).

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

Практическая №9, вариант 1:

```bash
PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab09_ensemble_learning
```

Готовый [отчёт №9](results/lab09_ensemble_learning/report.md) включает описание
данных, предобработку, код моделей, метрики, два графика и сравнительный анализ.
Исходное задание находится в `docs/assignments/practical_09.docx`,
предоставленный датасет — в `data/raw/archive.zip`.

Проверки:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Все сценарии используют фиксированный `random_state=42`, backend `Agg` для
графиков и при повторном запуске обновляют только собственный каталог в
`results/`.
