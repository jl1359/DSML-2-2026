import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_phiusiil_sample, load_phiusiil_processed

SAVED_DIR = os.path.join(BASE_DIR, "models", "unsupervised", "guardados")
RESULTS_DIR = os.path.join(BASE_DIR, "models", "unsupervised", "resultados")
os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

TARGET_COL = "label"
EXCLUDE_COLS = [TARGET_COL, "URL", "Domain", "TLD", "Title", "FILENAME"]


def extract_features(df):
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in EXCLUDE_COLS
    ]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy() if TARGET_COL in df.columns else None
    urls = df["URL"].to_numpy() if "URL" in df.columns else None
    return X, y, urls, feature_cols


def evaluate_elbow_and_silhouette(X, k_values=range(2, 9), silhouette_sample_size=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    sub_size = min(silhouette_sample_size, n_samples)
    sub_idx = rng.choice(n_samples, size=sub_size, replace=False)
    X_sub = X[sub_idx]

    records = []
    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        model.fit(X)
        inertia = float(model.inertia_)

        sub_model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        sub_labels = sub_model.fit_predict(X_sub)
        sil = float(silhouette_score(X_sub, sub_labels))

        records.append({
            "k": int(k),
            "inertia": inertia,
            "silhouette": sil,
        })

    return pd.DataFrame(records)


def plot_elbow_curve(elbow_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_title("Metodo del Codo (Inercia Intra-Cluster)", fontweight="bold")
    axes[0].set_xlabel("Numero de Clusters (k)")
    axes[0].set_ylabel("Inercia (WCSS)")
    axes[0].set_xticks(elbow_df["k"])

    axes[1].plot(elbow_df["k"], elbow_df["silhouette"], marker="s", linewidth=2, color="#2ca02c")
    axes[1].set_title("Coeficiente de Silueta Promedio", fontweight="bold")
    axes[1].set_xlabel("Numero de Clusters (k)")
    axes[1].set_ylabel("Score de Silueta")
    axes[1].set_xticks(elbow_df["k"])

    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_centroid_distances(X, model, labels):
    centroids = model.cluster_centers_
    all_distances = model.transform(X)

    assigned_dist = all_distances[np.arange(len(X)), labels]

    masked_distances = all_distances.copy()
    masked_distances[np.arange(len(X)), labels] = np.inf
    nearest_other_dist = np.min(masked_distances, axis=1)
    margin_to_other = nearest_other_dist - assigned_dist

    centroid_pair_dist = np.linalg.norm(
        centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=-1
    )

    dist_df = pd.DataFrame({
        "cluster": labels,
        "dist_centroide_asignado": assigned_dist,
        "dist_otro_centroide_mas_cercano": nearest_other_dist,
        "margen_separacion": margin_to_other,
    })

    cluster_summary = dist_df.groupby("cluster").agg(
        cantidad=("dist_centroide_asignado", "count"),
        dist_media=("dist_centroide_asignado", "mean"),
        dist_mediana=("dist_centroide_asignado", "median"),
        dist_std=("dist_centroide_asignado", "std"),
        dist_max=("dist_centroide_asignado", "max"),
        margen_medio=("margen_separacion", "mean"),
    ).reset_index()

    return {
        "assigned_dist": assigned_dist,
        "nearest_other_dist": nearest_other_dist,
        "margin_to_other": margin_to_other,
        "all_distances": all_distances,
        "centroid_pair_dist": centroid_pair_dist,
        "cluster_summary": cluster_summary,
    }


def plot_distance_analysis(dist_info, output_path):
    assigned_dist = dist_info["assigned_dist"]
    pair_dist = dist_info["centroid_pair_dist"]
    n_clusters = len(pair_dist)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for c in range(n_clusters):
        mask = (dist_info["all_distances"].argmin(axis=1) == c)
        axes[0].hist(
            assigned_dist[mask],
            bins=50,
            alpha=0.6,
            label=f"Cluster {c}",
        )

    axes[0].set_title("Distribucion de Distancias al Centroide Asignado", fontweight="bold")
    axes[0].set_xlabel("Distancia Euclidiana al Centroide")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()

    sns.heatmap(
        pair_dist,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=[f"C{i}" for i in range(n_clusters)],
        yticklabels=[f"C{i}" for i in range(n_clusters)],
        ax=axes[1],
    )
    axes[1].set_title("Distancia Euclidiana entre Centroides", fontweight="bold")

    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_cluster_purity(y_true, labels):
    ct = pd.crosstab(labels, y_true)
    majority_sum = ct.max(axis=1).sum()
    purity = float(majority_sum / len(y_true))
    return purity, ct


def run_kmeans_pipeline(use_processed_train=True, k_optimal=2, random_state=42):
    if use_processed_train:
        train_df, _ = load_phiusiil_processed()
        df = train_df
    else:
        df = load_phiusiil_sample()

    X, y_true, urls, feature_cols = extract_features(df)

    elbow_df = evaluate_elbow_and_silhouette(X, k_values=range(2, 9), random_state=random_state)
    elbow_plot_path = os.path.join(RESULTS_DIR, "kmeans_metodo_codo_silueta.png")
    plot_elbow_curve(elbow_df, elbow_plot_path)
    elbow_csv_path = os.path.join(RESULTS_DIR, "kmeans_evaluacion_k.csv")
    elbow_df.to_csv(elbow_csv_path, index=False)

    kmeans = KMeans(n_clusters=k_optimal, init="k-means++", n_init=15, max_iter=300, random_state=random_state)
    labels = kmeans.fit_predict(X)

    dist_info = analyze_centroid_distances(X, kmeans, labels)
    dist_plot_path = os.path.join(RESULTS_DIR, "kmeans_analisis_distancias.png")
    plot_distance_analysis(dist_info, dist_plot_path)
    dist_csv_path = os.path.join(RESULTS_DIR, "kmeans_resumen_distancias.csv")
    dist_info["cluster_summary"].to_csv(dist_csv_path, index=False)

    top_k = min(20, len(X))
    far_idx = np.argsort(dist_info["assigned_dist"])[::-1][:top_k]
    far_records = []
    for rank, idx in enumerate(far_idx):
        row = {
            "rank": rank,
            "sample_index": int(idx),
            "cluster": int(labels[idx]),
            "dist_centroide": float(dist_info["assigned_dist"][idx]),
            "margen_separacion": float(dist_info["margin_to_other"][idx]),
        }
        if urls is not None:
            row["url"] = str(urls[idx])
        if y_true is not None:
            row["label_real"] = int(y_true[idx])
        far_records.append(row)

    far_df = pd.DataFrame(far_records)
    far_csv_path = os.path.join(RESULTS_DIR, "kmeans_urls_mas_alejadas.csv")
    far_df.to_csv(far_csv_path, index=False)

    metrics = {
        "k": int(k_optimal),
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "inertia": float(kmeans.inertia_),
        "distancia_media_global": float(np.mean(dist_info["assigned_dist"])),
        "distancia_mediana_global": float(np.median(dist_info["assigned_dist"])),
        "distancia_max_global": float(np.max(dist_info["assigned_dist"])),
    }

    if y_true is not None:
        purity, ct = compute_cluster_purity(y_true, labels)
        ari = float(adjusted_rand_score(y_true, labels))
        homog = float(homogeneity_score(y_true, labels))
        metrics.update({
            "pureza": purity,
            "ari": ari,
            "homogeneidad": homog,
        })
        ct.to_csv(os.path.join(RESULTS_DIR, "kmeans_tabla_contingencia.csv"))

    model_path = os.path.join(SAVED_DIR, "modelo_kmeans.pkl")
    joblib.dump(kmeans, model_path)

    metrics_path = os.path.join(RESULTS_DIR, "kmeans_metricas.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[OK] k-Means completado.")
    print(f"[INFO] Modelo guardado en: {model_path}")
    print(f"[INFO] Metricas guardadas en: {metrics_path}")
    for k_metric, val in metrics.items():
        print(f"  - {k_metric}: {val}")

    return kmeans, labels, dist_info, metrics


if __name__ == "__main__":
    run_kmeans_pipeline(use_processed_train=True, k_optimal=2, random_state=42)
