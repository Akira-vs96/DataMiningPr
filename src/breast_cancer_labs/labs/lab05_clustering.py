"""Lab 05: unsupervised clustering of standardized WDBC features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.data import FEATURE_NAMES, RANDOM_STATE, load_clean_dataset
from breast_cancer_labs.plotting import apply_style, save_figure


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    truth: np.ndarray,
    algorithm: str,
    parameters: str,
) -> dict[str, object]:
    labels = np.asarray(labels)
    valid = labels != -1
    valid_labels = labels[valid]
    cluster_count = len(np.unique(valid_labels)) if valid.any() else 0
    row: dict[str, object] = {
        "algorithm": algorithm,
        "parameters": parameters,
        "clusters": cluster_count,
        "noise_points": int((~valid).sum()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "adjusted_rand_vs_diagnosis": adjusted_rand_score(truth, labels),
    }
    if cluster_count >= 2 and valid.sum() > cluster_count:
        row.update(
            {
                "silhouette": silhouette_score(X[valid], valid_labels),
                "davies_bouldin": davies_bouldin_score(X[valid], valid_labels),
                "calinski_harabasz": calinski_harabasz_score(X[valid], valid_labels),
            }
        )
    return row


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab05_clustering")
    data = load_clean_dataset()
    X = StandardScaler().fit_transform(data[FEATURE_NAMES])
    truth = data["diagnosis_binary"].to_numpy()

    k_rows = []
    k_models: dict[int, tuple[KMeans, np.ndarray]] = {}
    all_rows: list[dict[str, object]] = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X)
        k_models[k] = (model, labels)
        evaluation = evaluate_clustering(X, labels, truth, "K-means", f"k={k}")
        all_rows.append(evaluation)
        k_rows.append(
            {"k": k, "inertia": model.inertia_, "silhouette": evaluation["silhouette"]}
        )

    k_table = pd.DataFrame(k_rows)
    k_table.to_csv(output_dir / "kmeans_diagnostics.csv", index=False)
    optimal_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
    kmeans_labels = k_models[optimal_k][1]

    agglomerative = AgglomerativeClustering(n_clusters=2, linkage="ward")
    agglomerative_labels = agglomerative.fit_predict(X)
    all_rows.append(
        evaluate_clustering(
            X,
            agglomerative_labels,
            truth,
            "Agglomerative",
            "n_clusters=2, linkage=ward",
        )
    )

    dbscan_labels: dict[float, np.ndarray] = {}
    for eps in [2.0, 2.5, 3.0, 3.5, 4.0]:
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        dbscan_labels[eps] = labels
        all_rows.append(
            evaluate_clustering(X, labels, truth, "DBSCAN", f"eps={eps}, min_samples=5")
        )

    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(output_dir / "clustering_comparison.csv", index=False)
    valid_dbscan = comparison[
        (comparison["algorithm"] == "DBSCAN") & comparison["silhouette"].notna()
    ]
    if valid_dbscan.empty:
        best_eps = 3.0
    else:
        best_parameters = valid_dbscan.loc[valid_dbscan["silhouette"].idxmax(), "parameters"]
        best_eps = float(str(best_parameters).split("=")[1].split(",")[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(k_table["k"], k_table["inertia"], marker="o", color="#264653")
    axes[0].set(xlabel="k", ylabel="Inertia", title="K-means elbow")
    axes[1].plot(k_table["k"], k_table["silhouette"], marker="o", color="#e76f51")
    axes[1].axvline(optimal_k, color="#2a9d8f", linestyle="--", label=f"best k={optimal_k}")
    axes[1].set(xlabel="k", ylabel="Silhouette", title="K-means silhouette")
    axes[1].legend()
    save_figure(fig, output_dir / "kmeans_diagnostics.png")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    panels = [
        ("Known diagnosis (reference)", truth),
        (f"K-means (k={optimal_k})", kmeans_labels),
        ("Agglomerative (k=2)", agglomerative_labels),
        (f"DBSCAN (eps={best_eps:g})", dbscan_labels[best_eps]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for axis, (title, labels) in zip(axes.flat, panels):
        axis.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=24, alpha=0.75)
        axis.set(title=title, xlabel="PC1", ylabel="PC2")
    fig.suptitle(
        f"PCA projection ({pca.explained_variance_ratio_.sum():.1%} variance explained)"
    )
    save_figure(fig, output_dir / "pca_cluster_comparison.png")

    candidates = comparison.dropna(subset=["silhouette"])
    best = candidates.loc[candidates["silhouette"].idxmax()]
    report = f"""# Лабораторная работа №5 — кластеризация

Все 30 признаков стандартизированы до кластеризации. Для K-means проверены
`k=2…6`; по максимальному silhouette выбран **k={optimal_k}**. Дополнительно
сравнены Agglomerative Clustering и DBSCAN с пятью значениями `eps`.

Лучший внутренний silhouette: **{best['silhouette']:.4f}** —
{best['algorithm']} ({best['parameters']}). Adjusted Rand Index с известным
диагнозом приведён только как внешняя диагностическая метрика: target не
участвовал в обучении кластеров.

PCA применяется только для двумерной визуализации; сами алгоритмы обучены на
полном стандартизированном 30-мерном пространстве.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(comparison.round(4).to_string(index=False))
    print(f"Optimal K-means k: {optimal_k}")


if __name__ == "__main__":
    main()
