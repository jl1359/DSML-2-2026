"""
Módulo de Preprocesamiento y Análisis Exploratorio de Datos (EDA).
Implementa la limpieza, escalado Z-Score, partición Train/Test y generación de gráficos.
"""
import os
import sys

# Asegurar que la raíz del proyecto esté en el sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_phiusiil_raw, RAW_DIR, PROCESSED_DIR
FIGURAS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models", "phiusiil", "unsupervised")

os.makedirs(FIGURAS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Estilo para las gráficas
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
        print("🗑️ Columna 'FILENAME' eliminada.")

    # Eliminar duplicados en 'URL'
    df = df.drop_duplicates(subset=['URL']).reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)
    print(f"🧹 Duplicados eliminados: {duplicates_removed} filas.")
    print(f"📊 Registros limpios resultantes: {len(df)}")
    return df

def generate_eda_plots(df):
    """
    Genera y guarda las gráficas oficiales para el Pre-Informe 1.
    """
    print("📈 Generando gráficas para el Pre-Informe 1...")
    
    # 1. Distribución de Clases (Legítimo vs Phishing)
    plt.figure(figsize=(7, 5))
    class_counts = df['label'].value_counts()
    labels = ['Legítimo (1)', 'Phishing (0)']
    values = [class_counts.get(1, 0), class_counts.get(0, 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    plt.title("Distribución de Clases en PhiUSIIL", fontsize=14, fontweight='bold', pad=15)
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
    print(f"  ✅ Guardada: {fig_path1}")

    # 2. Distribución de Longitud de URL (Legítimo vs Phishing)
    plt.figure(figsize=(9, 5))
    # Filtramos para visualizar el 99% de los datos (sin outliers extremos que rompan el eje)
    limit_len = df['URLLength'].quantile(0.99)
    df_filtered = df[df['URLLength'] <= limit_len]
    
    sns.kdeplot(data=df_filtered, x='URLLength', hue='label', common_norm=False, 
                fill=True, palette={1: '#2ecc71', 0: '#e74c3c'}, alpha=0.4, linewidth=2)
    plt.title("Densidad de Longitud de URL (Legítimo vs Phishing)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Longitud de la URL (caracteres)", fontsize=12)
    plt.ylabel("Densidad", fontsize=12)
    plt.legend(title="Clase", labels=['Legítimo (1)', 'Phishing (0)'])
    fig_path2 = os.path.join(FIGURAS_DIR, "02_distribucion_longitud_url.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  ✅ Guardada: {fig_path2}")

    # 3. Matriz de Correlación de Características Numéricas Clave
    plt.figure(figsize=(10, 8))
    # Seleccionamos las variables numéricas más correlacionadas con 'label'
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['label'].abs().sort_values(ascending=False)
    top_features = correlations.head(12).index.tolist()
    
    corr_matrix = numeric_df[top_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Matriz de Correlación (Top 12 Características)", fontsize=14, fontweight='bold', pad=15)
    fig_path3 = os.path.join(FIGURAS_DIR, "03_matriz_correlacion.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"  ✅ Guardada: {fig_path3}")

    # 4. Indicadores de Ciberseguridad (Uso de HTTPS, Presencia de Password, IP en Dominio)
    plt.figure(figsize=(9, 5))
    indicators = ['IsHTTPS', 'IsDomainIP', 'HasPasswordField', 'HasExternalFormSubmit']
    available_indicators = [ind for ind in indicators if ind in df.columns]
    
    ind_data = []
    for ind in available_indicators:
        leg_pct = df[df['label'] == 1][ind].mean() * 100
        phish_pct = df[df['label'] == 0][ind].mean() * 100
        ind_data.append({'Indicador': ind, 'Legítimo': leg_pct, 'Phishing': phish_pct})
        
    ind_df = pd.DataFrame(ind_data).melt(id_vars='Indicador', var_name='Tipo', value_name='Porcentaje')
    sns.barplot(data=ind_df, x='Indicador', y='Porcentaje', hue='Tipo', palette={'Legítimo': '#2ecc71', 'Phishing': '#e74c3c'})
    plt.title("Presencia de Indicadores Clave en URLs (% de Sitios)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("% de Presencia", fontsize=12)
    plt.xlabel("")
    fig_path4 = os.path.join(FIGURAS_DIR, "04_indicadores_seguridad.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"  ✅ Guardada: {fig_path4}")

def preprocess_and_split(df, test_size=0.2, random_state=42):
    """
    1. Separa variables numéricas y variables de texto.
    2. Divide estratificadamente en Train (80%) y Test (20%).
    3. Aplica estandarización Z-score a las variables numéricas.
    4. Guarda los archivos en data/phiusiil/processed/.
    """
    print("\n⚙️ Iniciando partición y escalado de datos...")
    
    # Identificar columnas numéricas (excluyendo label)
    # y columnas de texto (URL es la primordial para Deep Learning)
    text_col = 'URL'
    target_col = 'label'
    
    # Identificamos columnas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()
    print(f"🔢 Total de características numéricas para No Supervisado: {len(num_cols)}")
    
    # División Train/Test estratificada
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    
    print(f"📦 Train set: {len(train_df):,} filas ({(1-test_size)*100:.0f}%)")
    print(f"📦 Test set:  {len(test_df):,} filas ({test_size*100:.0f}%)")
    
    # Escalado Z-Score (Diapositiva 1.10 de la docente)
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df_scaled[num_cols] = scaler.transform(test_df[num_cols])
    
    # Guardar el scaler para usarlo en la inferencia / Streamlit
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler_zscore.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Escalador Z-Score guardado en: {scaler_path}")
    
    # Guardar train.csv y test.csv procesados
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    
    train_df_scaled.to_csv(train_path, index=False)
    test_df_scaled.to_csv(test_path, index=False)
    
    print(f"💾 Conjunto de entrenamiento guardado en: {train_path}")
    print(f"💾 Conjunto de prueba guardado en: {test_path}")
    print("✅ Preprocesamiento completado exitosamente.")

def run_pipeline():
    """Ejecuta el pipeline completo de la Etapa 1."""
    print("🚀 Iniciando Etapa 1: Limpieza, EDA y Preprocesamiento...")
    df = load_phiusiil_raw()
    df_clean = clean_data(df)
    generate_eda_plots(df_clean)
    preprocess_and_split(df_clean)

if __name__ == "__main__":
    run_pipeline()
