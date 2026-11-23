"""
Modulo de Reduccion de Dimensionalidad y Proyeccion Espacial con PCA (Principal Component Analysis).
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

def ejecutar_pca():
    print("[INFO] Cargando datos procesados para analisis PCA...")
    train_df, test_df = load_phiusiil_processed()

    TARGET = "label"
    EXCLUDE_COLS = [TARGET, "URL"]
    FEATURE_COLS = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]

    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET].values

    # 1. Ajuste de PCA con 10 componentes para analizar varianza explicada acumulada
    pca_10 = PCA(n_components=10, random_state=42)
    pca_10.fit(X)

    var_indiv = pca_10.explained_variance_ratio_
    var_acum = np.cumsum(var_indiv)

    print("\n=== VARIANZA EXPLICADA POR COMPONENTES PRINCIPALES ===")
    for i in range(len(var_indiv)):
        print(f"Componente {i+1:2d}: {var_indiv[i]*100:5.2f}% | Acumulada: {var_acum[i]*100:5.2f}%")

    # Grafica de Varianza Explicada (Scree Plot)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    comp_range = range(1, 11)

    bars = ax1.bar(comp_range, var_indiv * 100, color="#3498db", alpha=0.7, label="Varianza Individual (%)")
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("Varianza Explicada Individual (%)", color="#2980b9")
    ax1.set_xticks(comp_range)

    ax2 = ax1.twinx()
    line = ax2.plot(comp_range, var_acum * 100, color="#e74c3c", marker="o", lw=2, label="Varianza Acumulada (%)")
    ax2.set_ylabel("Varianza Explicada Acumulada (%)", color="#c0392b")
    ax2.grid(False)

    plt.title("Analisis de Varianza Explicada por PCA (Primeros 10 Componentes)", fontweight="bold", pad=12)
    
    scree_path = os.path.join(RESULTADOS_DIR, "14_pca_varianza_explicada.png")
    scree_doc_path = os.path.join(DOCS_FIGS_DIR, "14_pca_varianza_explicada.png")
    plt.savefig(scree_path, dpi=300)
    plt.savefig(scree_doc_path, dpi=300)
    plt.close()
    print(f"[OK] Scree Plot guardado en: {scree_path}")

    # 2. Reduccion a 2 Componentes para Visualizacion Espacial
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X)

    print(f"[INFO] Varianza explicada por PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%")
    print(f"[INFO] Varianza explicada por PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%")
    print(f"[INFO] Varianza total en 2D:      {np.sum(pca_2d.explained_variance_ratio_)*100:.2f}%")

    # Guardar modelo PCA 2D
    pca_path = os.path.join(GUARDADOS_DIR, "pca_2d.joblib")
    joblib.dump(pca_2d, pca_path)
    print(f"[OK] Modelo PCA 2D guardado en: {pca_path}")

    # Cargar clusters de K-Means para graficar
    kmeans_path = os.path.join(GUARDADOS_DIR, "kmeans_k2.joblib")
    if os.path.exists(kmeans_path):
        kmeans_k2 = joblib.load(kmeans_path)
        cluster_labels = kmeans_k2.labels_
    else:
        from sklearn.cluster import KMeans
        kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        cluster_labels = kmeans_k2.labels_

    # Submuestra para renderizado grafico nitido y eficiente (20,000 puntos)
    np.random.seed(42)
    vis_idx = np.random.choice(len(X), size=20000, replace=False)
    X_pca_vis = X_pca[vis_idx]
    y_vis = y[vis_idx]
    clusters_vis = cluster_labels[vis_idx]

    # Grafica Comparativa 2D: Clases Reales vs Clusters No Supervisados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Clases Reales
    scatter1 = ax1.scatter(
        X_pca_vis[y_vis == 0, 0], X_pca_vis[y_vis == 0, 1],
        c="#2980b9", alpha=0.3, s=8, label="Legitimo (Clase 0)"
    )
    scatter2 = ax1.scatter(
        X_pca_vis[y_vis == 1, 0], X_pca_vis[y_vis == 1, 1],
        c="#e74c3c", alpha=0.3, s=8, label="Phishing (Clase 1)"
    )
    ax1.set_title("Proyeccion PCA 2D: Etiquetas Reales (Ground Truth)", fontweight="bold")
    ax1.set_xlabel(f"Componente Principal 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"Componente Principal 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.legend(loc="upper right", markerscale=3)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Agrupamiento No Supervisado (K-Means)
    ax2.scatter(
        X_pca_vis[clusters_vis == 0, 0], X_pca_vis[clusters_vis == 0, 1],
        c="#27ae60", alpha=0.3, s=8, label="Cluster 0 (K-Means)"
    )
    ax2.scatter(
        X_pca_vis[clusters_vis == 1, 0], X_pca_vis[clusters_vis == 1, 1],
        c="#8e44ad", alpha=0.3, s=8, label="Cluster 1 (K-Means)"
    )
    ax2.set_title("Proyeccion PCA 2D: Agrupamiento No Supervisado (K-Means k=2)", fontweight="bold")
    ax2.set_xlabel(f"Componente Principal 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"Componente Principal 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.legend(loc="upper right", markerscale=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pca_fig_path = os.path.join(RESULTADOS_DIR, "12_pca_2d_clusters_kmeans.png")
    pca_doc_path = os.path.join(DOCS_FIGS_DIR, "12_pca_2d_clusters_kmeans.png")
    plt.savefig(pca_fig_path, dpi=300)
    plt.savefig(pca_doc_path, dpi=300)
    plt.close()
    print(f"[OK] Visualizacion PCA 2D guardada en: {pca_fig_path}")

    # Guardar Metricas de PCA
    metricas_pca = {
        "algoritmo": "PCA",
        "n_componentes_evaluados": 10,
        "varianza_pc1": float(pca_2d.explained_variance_ratio_[0]),
        "varianza_pc2": float(pca_2d.explained_variance_ratio_[1]),
        "varianza_total_2d": float(np.sum(pca_2d.explained_variance_ratio_)),
        "varianza_acumulada_10": [float(v) for v in var_acum]
    }
    with open(os.path.join(RESULTADOS_DIR, "metricas_pca.json"), "w", encoding="utf-8") as f:
        json.dump(metricas_pca, f, indent=4)

    return pca_2d, X_pca

if __name__ == "__main__":
    ejecutar_pca()
