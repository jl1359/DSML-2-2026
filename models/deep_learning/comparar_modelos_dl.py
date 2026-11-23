"""
Comparacion y Analisis de Rendimiento de los 3 Modelos de Deep Learning (MLP, CNN-1D, LSTM).
Genera tabla comparativa y graficos consolidados para el Pre-Informe 1.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "models", "deep_learning", "resultados")
DOCS_FIGS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")

os.makedirs(DOCS_FIGS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def generar_comparativa():
    archivos = {
        "MLP (Perceptron Multicapa)": os.path.join(RESULTS_DIR, "metricas_mlp.json"),
        "CNN-1D (Convolucional)": os.path.join(RESULTS_DIR, "metricas_cnn.json"),
        "LSTM (Recurrente)": os.path.join(RESULTS_DIR, "metricas_lstm.json")
    }

    datos = []
    for nombre, ruta in archivos.items():
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                d = json.load(f)
                datos.append({
                    "Modelo": nombre,
                    "Accuracy": d["accuracy"],
                    "Precision": d["precision"],
                    "Recall": d["recall_sensibilidad"],
                    "Especificidad": d["especificidad"],
                    "F1-Score": d["f1_score"],
                    "ROC-AUC": d["roc_auc"],
                    "Falsos Positivos": d["false_positives"],
                    "Falsos Negativos": d["false_negatives"]
                })

    if not datos:
        print("[ERROR] No se encontraron metricas de modelos para comparar.")
        return

    df_comp = pd.DataFrame(datos)
    csv_path = os.path.join(RESULTS_DIR, "tabla_comparativa_deep_learning.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"[OK] Tabla comparativa guardada: {csv_path}")

    # 1. Grafico Comparativo de Barras
    metricas = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_melt = pd.melt(df_comp, id_vars=["Modelo"], value_vars=metricas, var_name="Metrica", value_name="Puntaje")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melt, x="Metrica", y="Puntaje", hue="Modelo", palette=["#2980b9", "#27ae60", "#8e44ad"])
    plt.title("Comparativa de Rendimiento: Familias de Aprendizaje Profundo (Test Set)", fontweight="bold", pad=15)
    plt.ylim(0.95, 1.002)
    plt.ylabel("Puntaje Normalizado [0 - 1]")
    plt.legend(loc="lower right")

    comp_plot_path = os.path.join(DOCS_FIGS_DIR, "09_comparativa_modelos_dl.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"[OK] Grafico comparativo guardado: {comp_plot_path}")

    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE MODELOS DE APRENDIZAJE PROFUNDO (TEST SET):")
    print(df_comp.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    generar_comparativa()
