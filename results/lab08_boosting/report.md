# Лабораторная работа №8 — Gradient Boosting и XGBoost

В классификации участвуют **569** объектов и **30**
признаков. Использовано стратифицированное разбиение 70/30 с
random_state=42; положительный класс — malignant.

По ROC-AUC лучший результат показал **Gradient Boosting**:
ROC-AUC=0.9971, accuracy=0.9649,
recall malignant=0.9062. В медицинском контексте FN
(пропущенный malignant) особенно важен, поэтому accuracy рассматривается только
вместе с recall, F1, ROC-AUC и confusion matrix.

Пять наиболее используемых XGBoost признаков: worst_perimeter, worst_radius, mean_concave_points, worst_concave_points, mean_area. Feature importance
показывает использование признака моделью, но не доказывает причинность.
Эксперименты отдельно оценивают `n_estimators` и `max_depth`, включая разрыв
между train и test accuracy как индикатор переобучения.
