"""
Módulo de Preprocesamiento para el Dataset Secundario: Malicious Phish.
(Módulo privado / Clasificación Multiclase + Aprendizaje por Refuerzo)
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "malicious_phish")
RAW_PATH = os.path.join(DATA_DIR, "raw", "malicious_phish.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models", "malicious_phish")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Mapeo oficial de las 4 clases de amenazas
LABEL_MAP = {
    'benign': 0,
    'phishing': 1,
    'defacement': 2,
    'malware': 3
}

def clean_and_process_malicious(sample_per_class=10000, random_state=42):
    """
    1. Carga el dataset de Malicious Phish.
    2. Elimina URLs duplicadas.
    3. Mapea las categorías de amenazas a valores numéricos (0 a 3).
    4. Genera una muestra balanceada de desarrollo rápido.
    5. Divide en 80% Train y 20% Test y guarda los archivos.
    """
    print(f"📂 Cargando {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)
    initial_rows = len(df)
    print(f"📊 Registros iniciales: {initial_rows:,}")

    # Eliminar duplicados en 'url'
    df = df.drop_duplicates(subset=['url']).reset_index(drop=True)
    duplicates = initial_rows - len(df)
    print(f"🧹 URLs duplicadas eliminadas: {duplicates:,}")
    print(f"📊 Registros limpios únicos: {len(df):,}")

    # Mapeo de clases
    df['label'] = df['type'].map(LABEL_MAP)
    
    # Guardar el mapa de clases en JSON
    map_path = os.path.join(SAVED_MODELS_DIR, "label_mapping.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=4)
    print(f"💾 Mapeo de clases guardado en: {map_path}")

    # Crear muestra balanceada para desarrollo rápido (40.000 filas en total)
    print(f"\n📦 Generando muestra balanceada ({sample_per_class:,} por clase)...")
    sample_df = pd.concat([
        group.sample(n=min(len(group), sample_per_class), random_state=random_state)
        for _, group in df.groupby('label')
    ]).reset_index(drop=True)

    sample_path = os.path.join(SAMPLES_DIR, "malicious_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"💾 Muestra de {len(sample_df):,} filas guardada en: {sample_path}")

    # División Train / Test estratificada de la muestra
    train_df, test_df = train_test_split(
        sample_df, test_size=0.2, random_state=random_state, stratify=sample_df['label']
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"💾 Train set ({len(train_df):,} filas) guardado en: {train_path}")
    print(f"💾 Test set  ({len(test_df):,} filas) guardado en: {test_path}")
    print("✅ Preprocesamiento de Malicious Phish completado exitosamente.")

if __name__ == "__main__":
    clean_and_process_malicious()
