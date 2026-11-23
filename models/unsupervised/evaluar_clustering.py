"""
Modulo de Consolidacion y Evaluacion Comparativa de Modelos No Supervisados.
Materia: Machine Learning y Ciencia de los Datos (2026)
Docente: Lic. Patricia Rodriguez Bilbao
Area: Ciberseguridad / Deteccion de Phishing URLs
"""
import os
import sys
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTADOS_DIR = os.path.join(BASE_DIR, "models", "unsupervised", "resultados")

def consolidar_evaluacion_no_supervisada():
    kmeans_json = os.path.join(RESULTADOS_DIR, "metricas_kmeans.json")
    dbscan_json = os.path.join(RESULTADOS_DIR, "metricas_dbscan.json")
    pca_json = os.path.join(RESULTADOS_DIR, "metricas_pca.json")

    filas = []

    if os.path.exists(kmeans_json):
        with open(kmeans_json, "r", encoding="utf-8") as f:
            km = json.load(f)
            filas.append({
                "Enfoque / Modelo": "K-Means (k=2)",
                "Proposito": "Segmentacion Binaria (Legitimo vs Phishing)",
                "Metrica Primaria": f"Silueta: {km['silhouette_k2']:.4f}",
                "Pureza / Precision": f"{km['pureza_clustering']*100:.2f}%",
                "Indice de Rand (ARI)": f"{km['adjusted_rand_index']:.4f}",
                "Homogeneidad": f"{km['homogeneidad']:.4f}",
                "Observacion": "Agrupa geometricamente basandose en centroides."
            })

    if os.path.exists(dbscan_json):
        with open(dbscan_json, "r", encoding="utf-8") as f:
            db = json.load(f)
            filas.append({
                "Enfoque / Modelo": "DBSCAN (eps=3.0, min_samples=15)",
                "Proposito": "Deteccion de Anomalias y Ataques Zero-Day",
                "Metrica Primaria": f"Tasa Anomalias: {db['tasa_anomalias_porcentaje']:.2f}%",
                "Pureza / Precision": f"{db['precision_deteccion_amenazas_anomalas']:.2f}% (Phishing en Ruido)",
                "Indice de Rand (ARI)": "N/A (Metodo de Densidad)",
                "Homogeneidad": "N/A",
                "Observacion": "Aisla URLs atipicas que no siguen el patron denso."
            })

    if os.path.exists(pca_json):
        with open(pca_json, "r", encoding="utf-8") as f:
            pca = json.load(f)
            filas.append({
                "Enfoque / Modelo": "PCA (Reduccion 2D)",
                "Proposito": "Proyeccion Espacial y Separabilidad",
                "Metrica Primaria": f"Varianza 2D: {pca['varianza_total_2d']*100:.2f}%",
                "Pureza / Precision": "N/A",
                "Indice de Rand (ARI)": "N/A",
                "Homogeneidad": "N/A",
                "Observacion": f"PC1 ({pca['varianza_pc1']*100:.1f}%) + PC2 ({pca['varianza_pc2']*100:.1f}%)"
            })

    if filas:
        df = pd.DataFrame(filas)
        csv_path = os.path.join(RESULTADOS_DIR, "tabla_comparativa_clustering.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Tabla comparativa guardada en: {csv_path}")
        print("\n=== TABLA COMPARATIVA DE APRENDIZAJE NO SUPERVISADO ===")
        print(df.to_string(index=False))
        return df
    else:
        print("[ERROR] No se encontraron archivos de metricas para consolidar.")
        return None

if __name__ == "__main__":
    consolidar_evaluacion_no_supervisada()
