"""
Módulo para carga y preparación de datos.
Permite descomprimir automáticamente el dataset PhiUSIIL y cargar muestras ligeras.
"""
import os
import zipfile
import pandas as pd

# Rutas estándar del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Rutas de PhiUSIIL (Dataset principal del equipo)
PHIUSIIL_DIR = os.path.join(DATA_DIR, "phiusiil")
RAW_DIR = os.path.join(PHIUSIIL_DIR, "raw")
PROCESSED_DIR = os.path.join(PHIUSIIL_DIR, "processed")
SAMPLES_DIR = os.path.join(PHIUSIIL_DIR, "samples")

# Rutas de Malicious Phish (Dataset secundario / validación externa)
MALICIOUS_DIR = os.path.join(DATA_DIR, "malicious_phish")
MALICIOUS_RAW_DIR = os.path.join(MALICIOUS_DIR, "raw")

CSV_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
ZIP_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.zip"

def load_phiusiil_raw(nrows=None):
    """
    Carga el dataset completo de PhiUSIIL desde data/phiusiil/raw/.
    Si el archivo CSV no está descomprimido, lo extrae automáticamente del ZIP.
    """
    csv_path = os.path.join(RAW_DIR, CSV_FILENAME)
    zip_path = os.path.join(RAW_DIR, ZIP_FILENAME)

    if not os.path.exists(csv_path):
        if os.path.exists(zip_path):
            print(f"📦 Descomprimiendo {ZIP_FILENAME} en {RAW_DIR}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            print("✅ Descompresión completa.")
        else:
            raise FileNotFoundError(f"No se encontró ni el CSV ni el ZIP en {RAW_DIR}")

    print(f"📂 Cargando {CSV_FILENAME}...")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"📊 Dataset cargado con éxito: {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df

def create_and_save_sample(n_per_class=15000, random_state=42):
    """
    Crea una muestra ligera y balanceada en data/phiusiil/samples/ para pruebas rápidas.
    Por defecto toma 15,000 legítimas y 15,000 phishing (30,000 en total).
    """
    df = load_phiusiil_raw()
    
    # Muestra estratificada por la columna 'label'
    sample_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state)
    ).reset_index(drop=True)

    sample_path = os.path.join(SAMPLES_DIR, "phiusiil_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"💾 Muestra de {len(sample_df)} filas guardada en: {sample_path}")
    return sample_df

def load_phiusiil_sample():
    """
    Carga directamente la muestra ligera desde data/phiusiil/samples/.
    Si no existe, la genera automáticamente.
    """
    sample_path = os.path.join(SAMPLES_DIR, "phiusiil_sample.csv")
    if not os.path.exists(sample_path):
        return create_and_save_sample()
    return pd.read_csv(sample_path)

def load_phiusiil_processed():
    """
    Carga los conjuntos procesados train.csv y test.csv desde data/phiusiil/processed/.
    Si aún no existen, ejecuta automáticamente el pipeline de preprocesamiento.
    """
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("⚙️ Datos procesados no encontrados. Ejecutando pipeline de preprocesamiento...")
        from src.preprocessing import run_pipeline
        run_pipeline()

    print(f"📂 Cargando train.csv y test.csv desde {PROCESSED_DIR}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"✅ Datos listos: Train ({len(train_df):,} filas), Test ({len(test_df):,} filas).")
    return train_df, test_df

def load_malicious_phish_raw(nrows=None):
    """
    Carga el dataset de Malicious Phish desde data/malicious_phish/raw/ (si existe localmente).
    """
    csv_path = os.path.join(MALICIOUS_RAW_DIR, "malicious_phish.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo no encontrado en {csv_path}. Este dataset es privado/local.")
    return pd.read_csv(csv_path, nrows=nrows)

if __name__ == "__main__":
    df_sample = load_phiusiil_sample()
    print(f"Prueba de carga exitosa: {df_sample.shape}")
