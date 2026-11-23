
"""
Clustering Jerarquico Aglomerativo - Deteccion de Phishing
Materia: Machine Learning y Ciencia de los Datos (2026)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Rutas del proyecto
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_phiusiil_processed

RESULTADOS_DIR = os.path.join(
    BASE_DIR, "models", "unsupervised", "resultados"
)

DOCS_FIGS_DIR = os.path.join(
    BASE_DIR, "docs", "pre_informe_1", "figuras"
)

os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(DOCS_FIGS_DIR, exist_ok=True)


def ejecutar_jerarquico():

    print("[INFO] Cargando datos procesados...")
    train_df, test_df = load_phiusiil_processed()

    # Seleccionar una muestra reproducible
    muestra = train_df.sample(n=1000, random_state=42)

    # Separar caracteristicas y etiquetas reales
    columnas = [
        c for c in muestra.select_dtypes(include=[np.number]).columns
        if c != "label"
    ]

    X = muestra[columnas].values
    y = muestra["label"].values

    print(f"[OK] Muestras: {len(X)}")
    print(f"[OK] Caracteristicas: {len(columnas)}")

    
    print("\n=== CARACTERISTICAS UTILIZADAS ===")
    for i, columna in enumerate(columnas, start=1):
        print(f"{i}. {columna}")


    # 1. Calcular la jerarquia
    print("[INFO] Calculando agrupamiento jerarquico...")

    Z = linkage(X, method="ward")

    # 2. Generar dendrograma
    plt.figure(figsize=(12, 6))

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True
    )

    plt.title("Dendrograma - Clustering Jerarquico")
    plt.xlabel("Grupos de URL")
    plt.ylabel("Distancia de fusion")

    ruta_figura = os.path.join(
        RESULTADOS_DIR, "14_dendrograma_jerarquico.png"
    )

    plt.savefig(ruta_figura, dpi=300)
    plt.savefig(
        os.path.join(DOCS_FIGS_DIR, "14_dendrograma_jerarquico.png"),
        dpi=300
    )
    plt.close()

    print("[OK] Dendrograma generado.")


    # EXPERIMENTACION: probar diferentes cantidades de clusters
    print("\n=== EXPERIMENTACION JERARQUICA ===")

    resultados = []

    for k in range(2, 6):

        modelo_prueba = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )

        grupos_prueba = modelo_prueba.fit_predict(X)

        sil = silhouette_score(X, grupos_prueba)
        ari_prueba = adjusted_rand_score(y, grupos_prueba)

        tabla_prueba = pd.crosstab(
            pd.Series(y),
            pd.Series(grupos_prueba)
        )

        pureza_prueba = (
            tabla_prueba.max(axis=0).sum() / len(y)
        )

        resultados.append({
            "Clusters": k,
            "Silueta": round(sil, 4),
            "ARI": round(ari_prueba, 4),
            "Pureza": round(pureza_prueba, 4)
        })

    df_resultados = pd.DataFrame(resultados)

    print(df_resultados.to_string(index=False))

    df_resultados.to_csv(
        os.path.join(
            RESULTADOS_DIR,
            "experimentacion_jerarquico.csv"
        ),
        index=False
    )


    # 3. Aplicar clustering aglomerativo
    modelo = AgglomerativeClustering(
        n_clusters=2,
        linkage="ward"
    )

    grupos = modelo.fit_predict(X)

    # 4. Evaluacion
    silueta = silhouette_score(X, grupos)
    ari = adjusted_rand_score(y, grupos)
    homogeneidad = homogeneity_score(y, grupos)

    tabla = pd.crosstab(
        pd.Series(y, name="Clase real"),
        pd.Series(grupos, name="Cluster")
    )

    pureza = tabla.max(axis=0).sum() / len(y)

    print("\n=== RESULTADOS CLUSTERING JERARQUICO ===")
    print(tabla)
    print(f"Silueta: {silueta:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Homogeneidad: {homogeneidad:.4f}")
    print(f"Pureza: {pureza:.4f}")

    # 5. Guardar metricas
    metricas = {
        "algoritmo": "Clustering Jerarquico Aglomerativo",
        "n_muestras": len(X),
        "n_clusters": 2,
        "silhouette": float(silueta),
        "adjusted_rand_index": float(ari),
        "homogeneidad": float(homogeneidad),
        "pureza_clustering": float(pureza),
        "distribucion_clusters": {
            str(k): int(v)
            for k, v in zip(*np.unique(grupos, return_counts=True))
        }
    }

    ruta_json = os.path.join(
        RESULTADOS_DIR, "metricas_jerarquico.json"
    )

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4)

    print(f"\n[OK] Metricas guardadas: {ruta_json}")
    print(f"[OK] Grafico guardado: {ruta_figura}")

    return modelo, grupos, metricas


if __name__ == "__main__":
    ejecutar_jerarquico()
