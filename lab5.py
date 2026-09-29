from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).with_name('processed_iris.csv')
OUTPUT_DIR = Path(__file__).parent
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


def calculate_metrics(data, labels, algorithm, parameters):
    """Calculate clustering metrics, excluding DBSCAN noise points."""
    labels = np.asarray(labels)
    noise_count = int(np.sum(labels == -1))
    valid = labels != -1
    valid_data = data[valid]
    valid_labels = labels[valid]
    cluster_count = len(np.unique(valid_labels))

    if cluster_count < 2 or len(valid_data) <= cluster_count:
        return {
            'algorithm': algorithm,
            'parameters': parameters,
            'clusters': cluster_count,
            'noise_points': noise_count,
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
        }

    return {
        'algorithm': algorithm,
        'parameters': parameters,
        'clusters': cluster_count,
        'noise_points': noise_count,
        'silhouette': silhouette_score(valid_data, valid_labels),
        'davies_bouldin': davies_bouldin_score(valid_data, valid_labels),
        'calinski_harabasz': calinski_harabasz_score(valid_data, valid_labels),
    }


def main():
    # 1. Деректерді жүктеу және бастапқы төрт белгіні таңдау
    df = pd.read_csv(DATA_FILE)
    missing_columns = set(FEATURES) - set(df.columns)
    if missing_columns:
        raise ValueError(f'CSV файлында қажетті бағандар жоқ: {sorted(missing_columns)}')

    data = df[FEATURES].apply(pd.to_numeric, errors='coerce').dropna().copy()
    X = data.to_numpy()
    print('=== 1. Деректерді жүктеу ===')
    print(f'X.shape = {X.shape}')
    print(f'Белгілер: {FEATURES}')
    print(f'Missing values алынып тасталған жолдар саны: {len(df) - len(data)}')

    # 2. Масштабтау
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print('\n=== 2. StandardScaler ===')
    print('Масштабталған белгілердің орташа мәндері:', np.round(X_scaled.mean(axis=0), 4))
    print('Масштабталған белгілердің std мәндері:', np.round(X_scaled.std(axis=0), 4))

    # 3. K-means: k = 2...6 үшін Elbow және Silhouette
    k_values = range(2, 7)
    kmeans_results = []
    kmeans_metric_results = []
    kmeans_models = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        kmeans_models[k] = (model, labels)
        kmeans_results.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
        })
        kmeans_metric_results.append(
            calculate_metrics(X_scaled, labels, 'K-means', f'k={k}')
        )

    kmeans_table = pd.DataFrame(kmeans_results)
    optimal_k = int(kmeans_table.loc[kmeans_table['silhouette'].idxmax(), 'k'])
    kmeans_model, kmeans_labels = kmeans_models[optimal_k]
    print('\n=== 3. K-means: Elbow және Silhouette ===')
    print(kmeans_table.round(4).to_string(index=False))
    print(f'Таңдалған k = {optimal_k}: Silhouette ең жоғары мәнге ие.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(kmeans_table['k'], kmeans_table['inertia'], marker='o', color='#173f5f')
    axes[0].set_title('K-means Elbow')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(alpha=0.3)
    axes[1].plot(kmeans_table['k'], kmeans_table['silhouette'], marker='o', color='#d95f02')
    axes[1].axvline(optimal_k, color='#2ca25f', linestyle='--', label=f'optimal k={optimal_k}')
    axes[1].set_title('K-means Silhouette')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    kmeans_diagnostic_file = OUTPUT_DIR / 'lab5_kmeans_diagnostics.png'
    fig.savefig(kmeans_diagnostic_file, dpi=200)
    plt.close(fig)

    # 4. Agglomerative Clustering және DBSCAN параметрлерін зерттеу
    agglomerative_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    agglomerative_labels = agglomerative_model.fit_predict(X_scaled)

    dbscan_results = []
    dbscan_models = {}
    for eps in [0.3, 0.5, 0.7, 0.9]:
        parameters = f'eps={eps}, min_samples=5'
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X_scaled)
        dbscan_models[eps] = (model, labels)
        dbscan_results.append(calculate_metrics(X_scaled, labels, 'DBSCAN', parameters))

    agglomerative_result = calculate_metrics(
        X_scaled,
        agglomerative_labels,
        'Agglomerative',
        'n_clusters=3, linkage=ward',
    )
    results_table = pd.DataFrame([
        *kmeans_metric_results,
        agglomerative_result,
        *dbscan_results,
    ])

    print('\n=== 4. Agglomerative Clustering және DBSCAN ===')
    print('Agglomerative: n_clusters=3, linkage="ward"')
    print('DBSCAN параметрлері: eps = 0.3, 0.5, 0.7, 0.9; min_samples = 5')
    print(results_table.round(4).to_string(index=False))

    # 5. PCA 2D визуализациясы
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    visualization_models = [
        ('K-means', kmeans_labels),
        ('Agglomerative', agglomerative_labels),
        ('DBSCAN (eps=0.5)', dbscan_models[0.5][1]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for axis, (title, labels) in zip(axes, visualization_models):
        scatter = axis.scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=42, alpha=0.85
        )
        axis.set_title(title)
        axis.set_xlabel('PC1')
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('PC2')
    fig.suptitle('Iris кластерлерінің PCA 2D визуализациясы')
    fig.tight_layout()
    pca_file = OUTPUT_DIR / 'lab5_pca_clusters.png'
    fig.savefig(pca_file, dpi=200)
    plt.close(fig)

    # Барлық k мәндері үшін K-means PCA графиктері
    fig, axes = plt.subplots(1, len(k_values), figsize=(20, 4.2), sharex=True, sharey=True)
    for axis, k in zip(axes, k_values):
        axis.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=kmeans_models[k][1],
            cmap='viridis',
            s=30,
            alpha=0.85,
        )
        axis.set_title(f'K-means: k={k}')
        axis.set_xlabel('PC1')
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('PC2')
    fig.suptitle('K-means: барлық k мәндері үшін PCA 2D')
    fig.tight_layout()
    kmeans_all_pca_file = OUTPUT_DIR / 'lab5_pca_kmeans_all_k.png'
    fig.savefig(kmeans_all_pca_file, dpi=200)
    plt.close(fig)

    # Барлық eps мәндері үшін DBSCAN PCA графиктері
    eps_values = [0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(1, len(eps_values), figsize=(16, 4.2), sharex=True, sharey=True)
    for axis, eps in zip(axes, eps_values):
        axis.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=dbscan_models[eps][1],
            cmap='viridis',
            s=30,
            alpha=0.85,
        )
        axis.set_title(f'DBSCAN: eps={eps}')
        axis.set_xlabel('PC1')
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('PC2')
    fig.suptitle('DBSCAN: барлық eps мәндері үшін PCA 2D')
    fig.tight_layout()
    dbscan_all_pca_file = OUTPUT_DIR / 'lab5_pca_dbscan_all_eps.png'
    fig.savefig(dbscan_all_pca_file, dpi=200)
    plt.close(fig)

    # 6. Салыстыру кестесін сақтау және қорытынды
    results_file = OUTPUT_DIR / 'lab5_clustering_comparison.csv'
    results_table.to_csv(results_file, index=False)
    valid_results = results_table.dropna(subset=['silhouette']).copy()
    best_row = valid_results.loc[valid_results['silhouette'].idxmax()]
    print('\n=== 5. Қысқаша қорытынды ===')
    print(f'Ең жоғары Silhouette: {best_row["algorithm"]} ({best_row["parameters"]}), '
          f'{best_row["silhouette"]:.4f}.')
    print('Silhouette жоғары, Davies–Bouldin төмен және Calinski–Harabasz жоғары болса, '
          'кластерлер сапасы жақсы деп бағаланады.')
    print('Iris деректері үшін PCA графигінде класстарды ең жинақы және анық бөлген '
          'әдіс салыстыру кестесіндегі метрикалармен бірге негізделеді.')
    print(f'Сақталған файлдар: {kmeans_diagnostic_file.name}, {pca_file.name}, '
          f'{kmeans_all_pca_file.name}, {dbscan_all_pca_file.name}, {results_file.name}')


if __name__ == '__main__':
    main()