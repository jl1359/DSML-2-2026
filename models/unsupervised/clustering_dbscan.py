"""
Modulo de Aprendizaje No Supervisado: Deteccion de Anomalias y Ataques Zero-Day con DBSCAN.
Materia: Machine Learning y Ciencia de los Datos (2026)
Docente: Lic. Patricia Rodriguez Bilbao
Area: Ciberseguridad / Deteccion de Phishing URLs
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_phiusiil_processed

GUARDADOS_DIR = os.path.join(BASE_DIR, "models", "unsupervised", "guardados")
RESULTADOS_DIR = os.path.join(BASE_DIR, "models", "unsupervised", "resultados")
DOCS_FIGS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")

os.makedirs(GUARDADOS_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(DOCS_FIGS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def ejecutar_dbscan():
    print("[INFO] Cargando datos procesados para DBSCAN...")
    train_df, test_df = load_phiusiil_processed()

    TARGET = "label"
    EXCLUDE_COLS = [TARGET, "URL"]
    FEATURE_COLS = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]

    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET].values

    # Muestra representativa de 20,000 puntos para calibracion espacial
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=20000, replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    # 1. Calibracion de Epsilon (k-distance graph)
    k_nn = 10
    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X_sample)
    distances, indices = nbrs.kneighbors(X_sample)
    k_distances = np.sort(distances[:, k_nn - 1])

    # 2. Ejecucion de DBSCAN con parametros optimizados para ciberseguridad
    # eps=3.0, min_samples=15
    print("[INFO] Ejecutando DBSCAN (eps=3.0, min_samples=15)...")
    dbscan = DBSCAN(eps=3.0, min_samples=15, metric="euclidean")
    db_labels = dbscan.fit_predict(X_sample)

    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_anomalias = list(db_labels).count(-1)
    tasa_anomalias = (n_anomalias / len(db_labels)) * 100

    print(f"[OK] Clusters identificados: {n_clusters}")
    print(f"[OK] Puntos de Ruido / Anomalias detectadas (-1): {n_anomalias:,} ({tasa_anomalias:.2f}%)")

    # Analisis de las Anomalias vs Ground Truth
    idx_anomalias = np.where(db_labels == -1)[0]
    y_anomalos = y_sample[idx_anomalias]
    n_phishing_anomalo = int(np.sum(y_anomalos == 1))
    n_legitimo_anomalo = int(np.sum(y_anomalos == 0))
    precision_anomalia = (n_phishing_anomalo / n_anomalias) * 100 if n_anomalias > 0 else 0.0

    print(f"[SEGURIDAD] Anomalias que son Phishing (Ataques Raros / Zero-Day): {n_phishing_anomalo:,} ({precision_anomalia:.2f}%)")
    print(f"[SEGURIDAD] Anomalias que son Sitios Legitimos Raros:           {n_legitimo_anomalo:,}")

    # 3. Proyeccion PCA 2D para visualizacion de Anomalias
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)

    plt.figure(figsize=(10, 6))
    # Puntos normales en clusters
    plt.scatter(
        X_pca[db_labels != -1, 0], X_pca[db_labels != -1, 1],
        c="#2980b9", alpha=0.3, s=10, label=f"Patrones Densos Normales ({len(db_labels) - n_anomalias:,} URLs)"
    )
    # Puntos anomalos (Ruido / Zero-Day)
    plt.scatter(
        X_pca[db_labels == -1, 0], X_pca[db_labels == -1, 1],
        c="#e74c3c", alpha=0.7, s=25, marker="x", label=f"Anomalias / Ruido DBSCAN ({n_anomalias:,} URLs - Zero-Day)"
    )

    plt.title(f"Deteccion de Anomalias y Ataques Atipicos con DBSCAN (eps=3.0, min_samples=15)", fontweight="bold", pad=12)
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(loc="upper right", markerscale=1.5)
    plt.grid(True, alpha=0.3)

    dbscan_fig_path = os.path.join(RESULTADOS_DIR, "13_dbscan_deteccion_anomalias.png")
    dbscan_doc_path = os.path.join(DOCS_FIGS_DIR, "13_dbscan_deteccion_anomalias.png")
    plt.savefig(dbscan_fig_path, dpi=300)
    plt.savefig(dbscan_doc_path, dpi=300)
    plt.close()
    print(f"[OK] Visualizacion DBSCAN guardada en: {dbscan_fig_path}")

    # 4. Guardar Modelo y Metricas
    modelo_path = os.path.join(GUARDADOS_DIR, "dbscan_modelo.joblib")
    joblib.dump(dbscan, modelo_path)

    metricas_dbscan = {
        "algoritmo": "DBSCAN",
        "eps": 3.0,
        "min_samples": 15,
        "n_muestras_analizadas": len(db_labels),
        "n_clusters_densos": n_clusters,
        "n_anomalias_detectadas": n_anomalias,
        "tasa_anomalias_porcentaje": float(tasa_anomalias),
        "anomalias_phishing_confirmado": n_phishing_anomalo,
        "anomalias_legitimas_atipicas": n_legitimo_anomalo,
        "precision_deteccion_amenazas_anomalas": float(precision_anomalia)
    }

    with open(os.path.join(RESULTADOS_DIR, "metricas_dbscan.json"), "w", encoding="utf-8") as f:
        json.dump(metricas_dbscan, f, indent=4)

    return dbscan, db_labels, metricas_dbscan

if __name__ == "__main__":
    ejecutar_dbscan()
