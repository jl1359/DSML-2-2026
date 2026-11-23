"""
Entrenamiento y Evaluacion de Red Neuronal Convolucional 1D (CNN-1D) para Deteccion de Phishing.
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

MODELS_DIR = os.path.join(BASE_DIR, "models", "deep_learning", "guardados")
RESULTS_DIR = os.path.join(BASE_DIR, "models", "deep_learning", "resultados")
DOCS_FIGS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_FIGS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def train_and_evaluate_cnn():
    print("[INFO] Cargando datos procesados para CNN-1D...")
    train_df, test_df = load_phiusiil_processed()

    TARGET = "label"
    EXCLUDE_COLS = [TARGET, "URL"]
    FEATURE_COLS = [col for col in train_df.select_dtypes(include=[np.number]).columns if col not in EXCLUDE_COLS]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET].values

    # Reshape para Conv1D: (muestras, features, 1)
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"[INFO] Dimensiones X_train (CNN-1D): {X_train_cnn.shape}")
    print(f"[INFO] Dimensiones X_test  (CNN-1D): {X_test_cnn.shape}")

    # 1. Arquitectura CNN-1D
    print("[INFO] Construyendo arquitectura de Red Neuronal Convolucional 1D...")
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1], 1)),
        
        # Bloque Convolucional 1
        layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Bloque Convolucional 2
        layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Agrupamiento Global y Capas Densas
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )

    arch_file = os.path.join(MODELS_DIR, "arquitectura_cnn.txt")
    with open(arch_file, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"[OK] Arquitectura guardada en: {arch_file}")

    # 2. Callbacks y Entrenamiento
    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    print("[INFO] Iniciando entrenamiento de CNN-1D...")
    history = model.fit(
        X_train_cnn, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=512,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    print("[OK] Entrenamiento de CNN-1D completado.")

    # 3. Guardar Modelo .keras
    model_path = os.path.join(MODELS_DIR, "modelo_cnn_phishing.keras")
    model.save(model_path)
    print(f"[OK] Modelo CNN guardado en: {model_path}")

    # 4. Curvas de Aprendizaje
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history["loss"]) + 1)

    ax1.plot(epochs_range, history.history["loss"], label="Train Loss", linewidth=2, color="#2980b9")
    ax1.plot(epochs_range, history.history["val_loss"], label="Val Loss", linewidth=2, linestyle="--", color="#c0392b")
    ax1.set_title("Evolucion de Loss (CNN-1D)", fontweight="bold")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", linewidth=2, color="#27ae60")
    ax2.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", linewidth=2, linestyle="--", color="#f39c12")
    ax2.set_title("Evolucion de Accuracy (CNN-1D)", fontweight="bold")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    curvas_path = os.path.join(RESULTS_DIR, "05_curvas_aprendizaje_cnn.png")
    plt.savefig(curvas_path, dpi=300)
    plt.close()
    print(f"[OK] Curvas de aprendizaje CNN guardadas en: {curvas_path}")

    # 5. Evaluacion en Test
    print("[INFO] Evaluando CNN-1D sobre conjunto de prueba...")
    y_pred_probs = model.predict(X_test_cnn, batch_size=512).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

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
        "modelo": "Red Neuronal Convolucional 1D (CNN-1D)",
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

    metrics_file = os.path.join(RESULTS_DIR, "metricas_cnn.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[OK] Metricas JSON CNN guardadas en: {metrics_file}")

    print("\n" + "="*50)
    print("REPORTE EVALUACION CNN-1D EN TEST:")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"Precision:     {prec*100:.2f}%")
    print(f"Recall:        {rec*100:.2f}%")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("="*50 + "\n")

    # 6. Matriz de Confusion
    plt.figure(figsize=(7, 6))
    cm_labels = ["Phishing (0)", "Legitimo (1)"]
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Greens", xticklabels=cm_labels, yticklabels=cm_labels, linewidths=1, linecolor="black")
    plt.title(f"Matriz de Confusion (CNN-1D)\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}", fontweight="bold", pad=12)
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    cm_path = os.path.join(RESULTS_DIR, "06_matriz_confusion_cnn.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Matriz de confusion CNN guardada: {cm_path}")

    # 7. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#27ae60", linewidth=2.5, label=f"CNN-1D (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", linewidth=1.5, label="Aleatorio (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)", fontweight="bold")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)", fontweight="bold")
    plt.title("Curva ROC (CNN-1D)", fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    roc_path = os.path.join(RESULTS_DIR, "07_curva_roc_auc_cnn.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[OK] Curva ROC CNN guardada: {roc_path}")

if __name__ == "__main__":
    train_and_evaluate_cnn()
