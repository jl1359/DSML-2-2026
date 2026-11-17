"""
Módulo para carga y preparación de datos.
Permite descomprimir automáticamente el dataset y cargar muestras ligeras.
"""
import os
import zipfile
import pandas as pd

# Rutas estándar del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

CSV_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
ZIP_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.zip"

def load_phiusiil_raw(nrows=None):
    """
    Carga el dataset completo desde data/raw/.
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
    Crea una muestra ligera y balanceada en data/samples/ para pruebas rápidas.
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

if __name__ == "__main__":
    # Generar la muestra inicial para el equipo
    create_and_save_sample()
