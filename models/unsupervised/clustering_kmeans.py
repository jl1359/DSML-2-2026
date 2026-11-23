"""
Modulo de Aprendizaje No Supervisado: Agrupamiento con K-Means.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

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

def ejecutar_kmeans():
    print("[INFO] Cargando datos procesados para K-Means...")
    train_df, test_df = load_phiusiil_processed()

    TARGET = "label"
    EXCLUDE_COLS = [TARGET, "URL"]
    FEATURE_COLS = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]

    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET].values

    print(f"[INFO] Dimensiones de datos: {X.shape[0]:,} muestras, {X.shape[1]} caracteristicas.")

    # 1. Metodo del Codo (Elbow Method) y Coeficiente de Silueta
    k_range = range(2, 9)
    inercia = []
    siluetas = []

    # Submuestra estratificada para computo viable de silueta (15,000 muestras)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=15000, replace=False)
    X_sample = X[sample_indices]

    print("[INFO] Evaluando valores de k en el rango [2, 8]...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X)
        inercia.append(km.inertia_)

        # Silueta sobre la submuestra representativa
        km_sample = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_sample = km_sample.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels_sample)
        siluetas.append(sil)
        print(f"      k={k} -> Inercia: {km.inertia_:,.1f} | Silueta: {sil:.4f}")

    # Guardar Grafica 1: Metodo del Codo
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inercia, marker="o", color="#2980b9", lw=2, markersize=8)
    plt.title("Metodo del Codo (Elbow Method) - K-Means", fontweight="bold", pad=12)
    plt.xlabel("Numero de Grupos (k)")
    plt.ylabel("Inercia (Suma de Cuadrados Intra-Cluster / WCSS)")
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    
    codo_path = os.path.join(RESULTADOS_DIR, "10_metodo_codo_kmeans.png")
    codo_doc_path = os.path.join(DOCS_FIGS_DIR, "10_metodo_codo_kmeans.png")
    plt.savefig(codo_path, dpi=300)
    plt.savefig(codo_doc_path, dpi=300)
    plt.close()
    print(f"[OK] Grafica de Codo guardada en: {codo_path}")

    # Guardar Grafica 2: Coeficiente de Silueta
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, siluetas, marker="s", color="#27ae60", lw=2, markersize=8)
    plt.title("Coeficiente de Silueta segun Numero de Grupos (k)", fontweight="bold", pad=12)
    plt.xlabel("Numero de Grupos (k)")
    plt.ylabel("Silhouette Score Promedio")
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)

    sil_path = os.path.join(RESULTADOS_DIR, "11_silhouette_score_kmeans.png")
    sil_doc_path = os.path.join(DOCS_FIGS_DIR, "11_silhouette_score_kmeans.png")
    plt.savefig(sil_path, dpi=300)
    plt.savefig(sil_doc_path, dpi=300)
    plt.close()
    print(f"[OK] Grafica de Silueta guardada en: {sil_path}")

    # 2. Entrenamiento del Modelo Final con k=2 (Legitimo vs Phishing)
    print("[INFO] Ajustando modelo final K-Means con k=2...")
    kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=15, max_iter=300)
    cluster_labels = kmeans_k2.fit_predict(X)

    # Evaluar pureza y concordancia con Ground Truth
    ari = adjusted_rand_score(y, cluster_labels)
    homog = homogeneity_score(y, cluster_labels)
    compl = completeness_score(y, cluster_labels)
    v_meas = v_measure_score(y, cluster_labels)

    # Matriz de Contingencia (Clusters vs Clases Reales)
    tabla_contingencia = pd.crosstab(
        pd.Series(y, name="Clase Real (0=Legitimo, 1=Phishing)"),
        pd.Series(cluster_labels, name="Cluster Asignado (K-Means)")
    )
    print("\n=== TABLA DE CONTINGENCIA K-MEANS (k=2) ===")
    print(tabla_contingencia)

    # Calcular Pureza del Agrupamiento
    pureza = np.sum(np.amax(tabla_contingencia.values, axis=0)) / np.sum(tabla_contingencia.values)

    metricas_kmeans = {
        "algoritmo": "K-Means",
        "k_optimo": 2,
        "inercia_k2": float(kmeans_k2.inertia_),
        "silhouette_k2": float(siluetas[0]),
        "pureza_clustering": float(pureza),
        "adjusted_rand_index": float(ari),
        "homogeneidad": float(homog),
        "completitud": float(compl),
        "v_measure": float(v_meas),
        "distribucion_clusters": {
            "cluster_0": int(np.sum(cluster_labels == 0)),
            "cluster_1": int(np.sum(cluster_labels == 1))
        }
    }

    # Guardar Modelo Serializado
    modelo_path = os.path.join(GUARDADOS_DIR, "kmeans_k2.joblib")
    joblib.dump(kmeans_k2, modelo_path)
    print(f"[OK] Modelo K-Means serializado en: {modelo_path}")

    # Guardar Metricas
    json_path = os.path.join(RESULTADOS_DIR, "metricas_kmeans.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metricas_kmeans, f, indent=4)
    print(f"[OK] Metricas K-Means guardadas en: {json_path}")

    return kmeans_k2, cluster_labels, metricas_kmeans

if __name__ == "__main__":
    ejecutar_kmeans()
