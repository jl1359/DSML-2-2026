"""
Entrenamiento y Evaluacion de Red Neuronal Profunda (MLP) para Deteccion de Phishing.
Materia: Machine Learning y Ciencia de los Datos (2026)
Docente: Lic. Patricia Rodriguez Bilbao
Area: Seguridad | Dataset: PhiUSIIL
"""
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar path base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data_loader import load_phiusiil_processed

# Rutas de salida
MODELS_DIR = os.path.join(BASE_DIR, "models", "deep_learning", "guardados")
RESULTS_DIR = os.path.join(BASE_DIR, "models", "deep_learning", "resultados")
DOCS_FIGS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_FIGS_DIR, exist_ok=True)

# Configuracion de estilo grafico
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def train_and_evaluate():
    print("[INFO] Cargando conjuntos procesados train.csv y test.csv...")
    train_df, test_df = load_phiusiil_processed()

    TARGET = "label"
    EXCLUDE_COLS = [TARGET, "URL"]
    FEATURE_COLS = [col for col in train_df.select_dtypes(include=[np.number]).columns if col not in EXCLUDE_COLS]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET].values

    print(f"[INFO] Dimensiones X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[INFO] Dimensiones X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"[INFO] Total de caracteristicas predictoras: {len(FEATURE_COLS)}")

    # 1. Definicion de la Arquitectura Deep MLP
    print("[INFO] Construyendo arquitectura de Red Neuronal Profunda...")
    input_dim = X_train.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Capa oculta 1
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Capa oculta 2
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Capa oculta 3
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        
        # Capa de salida (Sigmoide para clasificacion binaria: 0 Phishing, 1 Legitimo)
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    # Guardar resumen de arquitectura en archivo de texto
    arch_file = os.path.join(MODELS_DIR, "arquitectura_mlp.txt")
    with open(arch_file, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"[OK] Arquitectura guardada en: {arch_file}")

    # 2. Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    # 3. Entrenamiento del Modelo
    print("[INFO] Iniciando entrenamiento de la Red Neuronal...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=512,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    print("[OK] Entrenamiento completado exitosamente.")

    # 4. Guardado del Modelo Entrenado (.keras)
    model_path = os.path.join(MODELS_DIR, "modelo_mlp_phishing.keras")
    model.save(model_path)
    print(f"[OK] Modelo serializado en: {model_path}")

    # 5. Generacion de Curvas de Aprendizaje (Loss y Accuracy)
    print("[INFO] Generando curvas de aprendizaje...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # Curva de Perdida
    ax1.plot(epochs_range, history.history["loss"], label="Train Loss", linewidth=2, color="#2980b9")
    ax1.plot(epochs_range, history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--", color="#c0392b")
    ax1.set_title("Evolucion de la Funcion de Perdida (Binary Crossentropy)", fontweight="bold")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Curva de Precision
    ax2.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", linewidth=2, color="#27ae60")
    ax2.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--", color="#f39c12")
    ax2.set_title("Evolucion de la Precision (Accuracy)", fontweight="bold")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    curvas_path = os.path.join(RESULTS_DIR, "01_curvas_aprendizaje_mlp.png")
    plt.savefig(curvas_path, dpi=300)
    plt.close()
    shutil.copy(curvas_path, os.path.join(DOCS_FIGS_DIR, "05_curvas_aprendizaje_mlp.png"))
    print(f"[OK] Curvas de aprendizaje guardadas: {curvas_path}")

    # 6. Predicciones en Conjunto de Prueba
    print("[INFO] Evaluando modelo sobre el conjunto de prueba independiente (Test)...")
    y_pred_probs = model.predict(X_test, batch_size=512).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Calculo de metricas formales
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    ap = average_precision_score(y_test, y_pred_probs)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    metrics_dict = {
        "modelo": "Red Neuronal Profunda (MLP)",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensibilidad": float(rec),
        "especificidad": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "total_test_samples": int(len(y_test)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    metrics_file = os.path.join(RESULTS_DIR, "metricas_mlp.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[OK] Metricas JSON guardadas en: {metrics_file}")

    print("\n" + "="*50)
    print("REPORTE DE EVALUACION EN CONJUNTO DE PRUEBA:")
    print(f"Accuracy:       {acc*100:.2f}%")
    print(f"Precision:      {prec*100:.2f}%")
    print(f"Recall:         {rec*100:.2f}%")
    print(f"Especificidad:  {specificity*100:.2f}%")
    print(f"F1-Score:       {f1:.4f}")
    print(f"ROC-AUC Score:  {auc:.4f}")
    print("="*50 + "\n")
    print(classification_report(y_test, y_pred, target_names=["Phishing (0)", "Legitimo (1)"], digits=4))

    # 7. Grafica de Matriz de Confusion
    plt.figure(figsize=(7, 6))
    cm_labels = ["Phishing (0)", "Legitimo (1)"]
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=cm_labels, yticklabels=cm_labels,
        linewidths=1, linecolor="black", cbar=True
    )
    plt.title(f"Matriz de Confusion (Test Set)\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}", fontweight="bold", pad=12)
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    cm_path = os.path.join(RESULTS_DIR, "02_matriz_confusion_mlp.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    shutil.copy(cm_path, os.path.join(DOCS_FIGS_DIR, "06_matriz_confusion_mlp.png"))
    print(f"[OK] Matriz de confusion guardada: {cm_path}")

    # 8. Curva ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980b9", linewidth=2.5, label=f"Red Neuronal MLP (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", linewidth=1.5, label="Clasificador Aleatorio (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)", fontweight="bold")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)", fontweight="bold")
    plt.title("Curva Caracteristica Operativa del Receptor (ROC)", fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    roc_path = os.path.join(RESULTS_DIR, "03_curva_roc_auc_mlp.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    shutil.copy(roc_path, os.path.join(DOCS_FIGS_DIR, "07_curva_roc_auc_mlp.png"))
    print(f"[OK] Curva ROC guardada: {roc_path}")

    # 9. Curva Precision-Recall
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, color="#27ae60", linewidth=2.5, label=f"MLP (Avg Precision = {ap:.4f})")
    plt.xlabel("Recall (Sensibilidad)", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.title("Curva Precision-Recall", fontweight="bold", pad=12)
    plt.legend(loc="lower left")
    pr_path = os.path.join(RESULTS_DIR, "04_precision_recall_mlp.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    shutil.copy(pr_path, os.path.join(DOCS_FIGS_DIR, "08_precision_recall_mlp.png"))
    print(f"[OK] Curva Precision-Recall guardada: {pr_path}")

    print("\n[OK] Pipeline de Deep Learning completado y validado.")

if __name__ == "__main__":
    train_and_evaluate()
