"""
Pipeline de Limpieza, Analisis Exploratorio (EDA) y Estandarizacion Z-Score.
Basado en el dataset PhiUSIIL para el proyecto academico de Machine Learning.
"""
import os
import sys
import joblib

# Configurar path base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_phiusiil_raw, RAW_DIR, PROCESSED_DIR
FIGURAS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "unsupervised", "guardados")

os.makedirs(FIGURAS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Estilo para las graficas
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def clean_data(df):
    """
    1. Elimina columnas irrelevantes como 'FILENAME'.
    2. Elimina URLs duplicadas para evitar fuga de datos.
    """
    initial_rows = len(df)
    
    # Eliminar FILENAME si existe
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
        print("[INFO] Columna 'FILENAME' eliminada.")

    # Eliminar duplicados en 'URL'
    df = df.drop_duplicates(subset=['URL']).reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)
    print(f"[INFO] Duplicados eliminados: {duplicates_removed} filas.")
    print(f"[INFO] Registros limpios resultantes: {len(df):,}")
    return df

def generate_eda_plots(df):
    """
    Genera y guarda las graficas oficiales para el Pre-Informe 1.
    """
    print("[INFO] Generando graficas para el Pre-Informe 1...")
    
    # 1. Distribucion de Clases (Legitimo vs Phishing)
    plt.figure(figsize=(7, 5))
    class_counts = df['label'].value_counts()
    labels = ['Legitimo (1)', 'Phishing (0)']
    values = [class_counts.get(1, 0), class_counts.get(0, 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    plt.title("Distribucion de Clases en PhiUSIIL", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Cantidad de URLs", fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.01), 
                 f"{yval:,} ({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0, max(values) * 1.12)
    fig_path1 = os.path.join(FIGURAS_DIR, "01_distribucion_clases.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"  [OK] Guardada: {fig_path1}")

    # 2. Distribucion de Longitud de URL (Legitimo vs Phishing)
    plt.figure(figsize=(9, 5))
    limit_len = df['URLLength'].quantile(0.99)
    df_filtered = df[df['URLLength'] <= limit_len]
    
    sns.kdeplot(data=df_filtered, x='URLLength', hue='label', common_norm=False, 
                fill=True, palette={1: '#2ecc71', 0: '#e74c3c'}, alpha=0.4, linewidth=2)
    plt.title("Densidad de Longitud de URL (Legitimo vs Phishing)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Longitud de la URL (caracteres)", fontsize=12)
    plt.ylabel("Densidad", fontsize=12)
    plt.legend(title="Clase", labels=['Legitimo (1)', 'Phishing (0)'])
    fig_path2 = os.path.join(FIGURAS_DIR, "02_distribucion_longitud_url.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  [OK] Guardada: {fig_path2}")

    # 3. Matriz de Correlacion de Caracteristicas Numericas Clave
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_label = numeric_df.corr()['label'].abs().sort_values(ascending=False)
    top_features = corr_with_label.head(15).index.tolist()
    
    corr_matrix = numeric_df[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Coeficiente de Correlacion de Pearson'})
    plt.title("Top 15 Variables mas Correlacionadas con Phishing", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    fig_path3 = os.path.join(FIGURAS_DIR, "03_matriz_correlacion.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"  [OK] Guardada: {fig_path3}")

    # 4. Analisis de Indicadores de Seguridad y Presencia de Certificados HTTPS
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'HttpsInHostname' in df.columns:
        sns.countplot(data=df, x='HttpsInHostname', hue='label', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
        axes[0].set_title("Presencia de 'https' en el Hostname", fontweight='bold')
        axes[0].set_xlabel("Https en Hostname (0: No, 1: Si)")
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend(labels=['Phishing (0)', 'Legitimo (1)'])
        
    if 'SymbolCount_Hyphen' in df.columns:
        sns.boxplot(data=df[df['SymbolCount_Hyphen'] < 10], x='label', y='SymbolCount_Hyphen', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
        axes[1].set_title("Cantidad de Guiones '-' en la URL", fontweight='bold')
        axes[1].set_xlabel("Clase (0: Phishing, 1: Legitimo)")
        axes[1].set_ylabel("Numero de Guiones")

    fig_path4 = os.path.join(FIGURAS_DIR, "04_indicadores_seguridad.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"  [OK] Guardada: {fig_path4}")
    print("[OK] Todas las figuras fueron generadas con exito.")

def preprocess_and_split(df, target_col='label', test_size=0.2, random_state=42):
    """
    Realiza la division estratificada Train/Test y estandariza las variables numericas
    usando Z-Score (StandardScaler) ajustado exclusivamente sobre Train.
    """
    print("[INFO] Iniciando division estratificada Train/Test y estandarizacion Z-Score...")
    
    # Identificamos columnas numericas
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()
    print(f"[INFO] Total de caracteristicas numericas escaladas: {len(num_cols)}")
    
    # Division Train/Test estratificada
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    
    print(f"[INFO] Train set: {len(train_df):,} filas ({(1-test_size)*100:.0f}%)")
    print(f"[INFO] Test set:  {len(test_df):,} filas ({test_size*100:.0f}%)")
    
    # Escalado Z-Score (Formula teorica: z = (x - mu) / sigma)
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df_scaled[num_cols] = scaler.transform(test_df[num_cols])
    
    # Guardar el scaler para usarlo en la inferencia / Streamlit
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler_zscore.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Escalador Z-Score guardado en: {scaler_path}")
    
    # Guardar train.csv y test.csv procesados
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    
    train_df_scaled.to_csv(train_path, index=False)
    test_df_scaled.to_csv(test_path, index=False)
    
    print(f"[OK] Conjunto de entrenamiento guardado en: {train_path}")
    print(f"[OK] Conjunto de prueba guardado en: {test_path}")
    print("[OK] Preprocesamiento completado exitosamente.")

def run_pipeline():
    """Ejecuta el pipeline completo de la Etapa 1."""
    print("[ETAPA 1] Limpieza, EDA y Preprocesamiento...")
    df = load_phiusiil_raw()
    df_clean = clean_data(df)
    generate_eda_plots(df_clean)
    preprocess_and_split(df_clean)

if __name__ == "__main__":
    run_pipeline()
