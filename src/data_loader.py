"""
Modulo para carga y preparacion de datos.
Permite descomprimir automaticamente el dataset PhiUSIIL y cargar muestras de desarrollo.
"""
import os
import zipfile
import pandas as pd

# Rutas estandar del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Rutas de PhiUSIIL (Dataset oficial)
PHIUSIIL_DIR = os.path.join(DATA_DIR, "phiusiil")
RAW_DIR = os.path.join(PHIUSIIL_DIR, "raw")
PROCESSED_DIR = os.path.join(PHIUSIIL_DIR, "processed")
SAMPLES_DIR = os.path.join(PHIUSIIL_DIR, "samples")

CSV_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
ZIP_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.zip"

def load_phiusiil_raw(nrows=None):
    """
    Carga el dataset completo de PhiUSIIL desde data/phiusiil/raw/.
    Si el archivo CSV no esta descomprimido, lo extrae automaticamente del ZIP.
    """
    csv_path = os.path.join(RAW_DIR, CSV_FILENAME)
    zip_path = os.path.join(RAW_DIR, ZIP_FILENAME)

    if not os.path.exists(csv_path):
        if os.path.exists(zip_path):
            print(f"[INFO] Descomprimiendo {ZIP_FILENAME} en {RAW_DIR}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            print("[OK] Descompresion completa.")
        else:
            raise FileNotFoundError(f"No se encontro ni el CSV ni el ZIP en {RAW_DIR}")

    print(f"[INFO] Cargando {CSV_FILENAME}...")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"[INFO] Dataset cargado: {df.shape[0]:,} filas y {df.shape[1]} columnas.")
    return df

def create_and_save_sample(n_per_class=15000, random_state=42):
    """
    Crea una muestra ligera y balanceada en data/phiusiil/samples/ para pruebas rapidas.
    Por defecto toma 15,000 legitimas y 15,000 phishing (30,000 en total).
    """
    df = load_phiusiil_raw()
    
    # Muestra estratificada por la columna 'label'
    sample_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state)
    ).reset_index(drop=True)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sample_path = os.path.join(SAMPLES_DIR, "phiusiil_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[OK] Muestra balanceada de {len(sample_df):,} filas guardada en: {sample_path}")
    return sample_df

def load_phiusiil_sample():
    """
    Carga directamente la muestra ligera desde data/phiusiil/samples/.
    Si no existe, la genera automaticamente.
    """
    sample_path = os.path.join(SAMPLES_DIR, "phiusiil_sample.csv")
    if not os.path.exists(sample_path):
        return create_and_save_sample()
    return pd.read_csv(sample_path)

def load_phiusiil_processed():
    """
    Carga los conjuntos procesados train.csv y test.csv desde data/phiusiil/processed/.
    Si aun no existen, ejecuta automaticamente el pipeline de preprocesamiento.
    """
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[INFO] Datos procesados no encontrados. Ejecutando pipeline de preprocesamiento...")
        from src.preprocessing import run_pipeline
        run_pipeline()

    print(f"[INFO] Cargando train.csv y test.csv desde {PROCESSED_DIR}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[OK] Datos listos: Train ({len(train_df):,} filas), Test ({len(test_df):,} filas).")
    return train_df, test_df

if __name__ == "__main__":
    df_sample = load_phiusiil_sample()
    print(f"[OK] Prueba de carga exitosa: {df_sample.shape}")
