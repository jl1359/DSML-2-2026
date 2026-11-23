"""
Interfaz Interactiva de Ciberseguridad: Detección y Mitigación de Phishing URLs
Materia: Machine Learning y Ciencia de los Datos (2026)
Universidad Mayor de San Simón - Cochabamba, Bolivia
Docente: Lic. Patricia Rodríguez Bilbao

Integra los tres paradigmas del proyecto:
1. Aprendizaje Profundo: Redes Neuronales (MLP, CNN-1D, LSTM) para predicción de alta precisión.
2. Aprendizaje No Supervisado: K-Means (k=2) y Proyección PCA 2D para validación de estructura geométrica.
3. Aprendizaje por Refuerzo: Agente Q-Learning para mitigación dinámica (Allow, Block, Sandbox).
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_phiusiil_processed

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Sistema de Ciberseguridad - Detección de Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS limpios
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1.2rem; border-radius: 8px; border-left: 5px solid #2563EB; }
    .status-badge-ok { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .status-badge-danger { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .status-badge-warn { background-color: #FEF08A; color: #713F12; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelos():
    """Carga los modelos entrenados y artefactos serializados."""
    import tensorflow as tf
    
    # 1. Modelos de Deep Learning
    path_mlp = os.path.join(BASE_DIR, "models", "deep_learning", "guardados", "modelo_mlp_phishing.keras")
    path_cnn = os.path.join(BASE_DIR, "models", "deep_learning", "guardados", "modelo_cnn_phishing.keras")
    
    modelo_mlp = tf.keras.models.load_model(path_mlp) if os.path.exists(path_mlp) else None
    modelo_cnn = tf.keras.models.load_model(path_cnn) if os.path.exists(path_cnn) else None
    
    # 2. Modelos de Clustering y Reducción
    path_kmeans = os.path.join(BASE_DIR, "models", "unsupervised", "guardados", "kmeans_k2.joblib")
    path_pca = os.path.join(BASE_DIR, "models", "unsupervised", "guardados", "pca_2d.joblib")
    
    modelo_kmeans = joblib.load(path_kmeans) if os.path.exists(path_kmeans) else None
    modelo_pca = joblib.load(path_pca) if os.path.exists(path_pca) else None
    
    # 3. Política Q-Learning
    path_qtable = os.path.join(BASE_DIR, "models", "reinforcement_learning", "guardados", "q_table_phishing.pkl")
    q_table = None
    if os.path.exists(path_qtable):
        with open(path_qtable, "rb") as f:
            q_table = pickle.load(f)
            
    return modelo_mlp, modelo_cnn, modelo_kmeans, modelo_pca, q_table


@st.cache_data
def cargar_muestra_datos():
    """Carga una muestra de datos de prueba para la evaluación interactiva."""
    _, df_test = load_phiusiil_processed()
    return df_test.sample(n=1000, random_state=42).reset_index(drop=True)


def discretizar_estado_rl(longitud, simbolos, https):
    """Mapea características continuas al espacio de 24 estados discretos del agente RL."""
    if longitud < -0.5:
        s_len = 0
    elif longitud < 0.2:
        s_len = 1
    elif longitud < 1.0:
        s_len = 2
    else:
        s_len = 3

    if simbolos < -0.2:
        s_sym = 0
    elif simbolos < 0.5:
        s_sym = 1
    else:
        s_sym = 2

    s_https = int(https > 0)
    return (s_len, s_sym, s_https)


def main():
    st.markdown("<div class='main-header'>Sistema Multi-Paradigma de Detección y Mitigación de Phishing URLs</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Trabajo Académico de Investigación | Machine Learning y Ciencia de los Datos 2026 | UMSS Cochabamba</div>", unsafe_allow_html=True)
    
    # Cargar modelos y datos
    with st.spinner("Cargando modelos neuronales, centroides de clustering y agente de refuerzo..."):
        modelo_mlp, modelo_cnn, modelo_kmeans, modelo_pca, q_table = cargar_modelos()
        df_muestra = cargar_muestra_datos()
    
    # Barra lateral de configuración
    st.sidebar.title("Configuración de Evaluación")
    
    modo_evaluacion = st.sidebar.radio(
        "Modo de Inspección:",
        ["Muestras Reales del Conjunto de Prueba", "Ajuste Manual de Parámetros"]
    )
    
    modelo_dl_sel = st.sidebar.selectbox(
        "Arquitectura de Deep Learning:",
        ["Perceptrón Multicapa (MLP)", "Red Convolucional 1D (CNN-1D)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Resumen de Modelos Activos")
    st.sidebar.info(
        "• Deep Learning: MLP / CNN-1D (99.996% Precisión)\n"
        "• Clustering: K-Means k=2 (98.51% Pureza Natural)\n"
        "• Reducción: PCA 2D (28.43% Varianza)\n"
        "• Refuerzo: Q-Learning MDP (Allow/Block/Sandbox)"
    )

    features_cols = [c for c in df_muestra.columns if c not in ["label", "url", "FILENAME"]]
    
    if modo_evaluacion == "Muestras Reales del Conjunto de Prueba":
        st.subheader("1. Selección de Muestra de Tráfico Web Real")
        
        col_filtro, col_idx = st.columns([1, 2])
        with col_filtro:
            filtro_clase = st.selectbox("Filtrar tipo de URL:", ["Todas", "Legítimas Reales (Clase 0)", "Phishing Reales (Clase 1)"])
        
        if filtro_clase == "Legítimas Reales (Clase 0)":
            df_filtrado = df_muestra[df_muestra["label"] == 0].reset_index(drop=True)
        elif filtro_clase == "Phishing Reales (Clase 1)":
            df_filtrado = df_muestra[df_muestra["label"] == 1].reset_index(drop=True)
        else:
            df_filtrado = df_muestra
            
        with col_idx:
            idx_seleccionado = st.slider("Índice de muestra a inspeccionar:", 0, len(df_filtrado) - 1, 0)
            
        muestra_actual = df_filtrado.iloc[idx_seleccionado]
        etiqueta_real = int(muestra_actual["label"])
        vector_x = muestra_actual[features_cols].values.reshape(1, -1).astype(np.float32)
        
        # Extracción de variables clave para visualización
        longitud_val = float(muestra_actual.get("URLLength", 0.0))
        simbolos_val = float(muestra_actual.get("CharContinuationRate", 0.0))
        https_val = float(muestra_actual.get("IsHTTPS", 0.0))
        
        st.write(f"**Etiqueta Real (Ground Truth):** {'🔴 SITIO DE PHISHING (1)' if etiqueta_real == 1 else '🟢 SITIO LEGÍTIMO (0)'}")

    else:
        st.subheader("1. Parámetros Estructurales de la URL (Valores Estandarizados Z-Score)")
        col1, col2, col3 = st.columns(3)
        with col1:
            longitud_val = st.slider("Longitud de la URL (URLLength):", -2.0, 5.0, 0.0, 0.1)
        with col2:
            simbolos_val = st.slider("Tasa de Continuación de Caracteres:", -2.0, 5.0, 0.0, 0.1)
        with col3:
            https_val = st.selectbox("Protocolo HTTPS:", [1.0, 0.0], format_func=lambda x: "Sí (Seguro)" if x == 1.0 else "No (Inseguro)")
            
        vector_x = np.zeros((1, len(features_cols)), dtype=np.float32)
        if "URLLength" in features_cols:
            vector_x[0, features_cols.index("URLLength")] = longitud_val
        if "CharContinuationRate" in features_cols:
            vector_x[0, features_cols.index("CharContinuationRate")] = simbolos_val
        if "IsHTTPS" in features_cols:
            vector_x[0, features_cols.index("IsHTTPS")] = https_val
            
        etiqueta_real = None

    # Pipeline Multi-Paradigma
    st.markdown("---")
    st.subheader("2. Veredictos del Pipeline Multi-Paradigma")
    
    col_dl, col_km, col_rl = st.columns(3)
    
    # A. Predicción Deep Learning
    with col_dl:
        st.markdown("#### A. Aprendizaje Profundo")
        modelo_activo = modelo_mlp if "MLP" in modelo_dl_sel else modelo_cnn
        if modelo_activo is not None:
            if "CNN" in modelo_dl_sel:
                prob = float(modelo_activo.predict(vector_x.reshape(1, len(features_cols), 1), verbose=0)[0, 0])
            else:
                prob = float(modelo_activo.predict(vector_x, verbose=0)[0, 0])
                
            pred_clase = 1 if prob >= 0.5 else 0
            
            st.metric(label="Probabilidad de Phishing", value=f"{prob * 100:.2f}%")
            if pred_clase == 1:
                st.error("Clasificación: PHISHING (MALICIOSA)")
            else:
                st.success("Clasificación: SITIO LEGÍTIMO")
        else:
            st.warning("Modelo Deep Learning no cargado.")

    # B. Agrupamiento K-Means y PCA
    with col_km:
        st.markdown("#### B. Aprendizaje No Supervisado")
        if modelo_kmeans is not None and modelo_pca is not None:
            cluster = int(modelo_kmeans.predict(vector_x)[0])
            coord_pca = modelo_pca.transform(vector_x)[0]
            
            cluster_nombre = "Cluster 0 (Perfil Malicioso)" if cluster == 0 else "Cluster 1 (Perfil Legítimo)"
            st.metric(label="Cluster Asignado por K-Means", value=f"Cluster {cluster}")
            st.info(f"{cluster_nombre}\n\nCoordenadas PCA: PC1={coord_pca[0]:.2f}, PC2={coord_pca[1]:.2f}")
        else:
            st.warning("Modelos no supervisados no disponibles.")

    # C. Mitigación Dinámica con Aprendizaje por Refuerzo
    with col_rl:
        st.markdown("#### C. Mitigación Dinámica (Q-Learning)")
        if q_table is not None:
            estado_discreto = discretizar_estado_rl(longitud_val, simbolos_val, https_val)
            q_valores = q_table.get(estado_discreto, [0.0, 0.0, 0.0])
            accion_optima = int(np.argmax(q_valores))
            
            nombres_acciones = ["PERMITIR (Allow)", "BLOQUEAR (Block)", "SANDBOX (Inspección Segura)"]
            
            st.metric(label="Acción Defensiva de la Política", value=nombres_acciones[accion_optima])
            if accion_optima == 0:
                st.success("Decisión: Conexión autorizada sin restricciones.")
            elif accion_optima == 1:
                st.error("Decisión: Enlace bloqueado inmediatamente en el cortafuegos.")
            else:
                st.warning("Decisión: Enlace aislado preventivamente para detonación segura.")
                
            st.caption(f"Valores Q: Allow={q_valores[0]:.1f} | Block={q_valores[1]:.1f} | Sandbox={q_valores[2]:.1f}")
        else:
            st.warning("Tabla Q-Learning no cargada.")

    # Visualización Espacial PCA
    st.markdown("---")
    st.subheader("3. Ubicación Geométrica de la URL en el Espacio PCA 2D")
    
    if modelo_pca is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Submuestra de fondo
        X_sub = df_muestra[features_cols].values[:300]
        pca_sub = modelo_pca.transform(X_sub)
        y_sub = df_muestra["label"].values[:300]
        
        ax.scatter(pca_sub[y_sub == 0, 0], pca_sub[y_sub == 0, 1], c="#2563EB", alpha=0.3, label="Legítimos (Fondo)")
        ax.scatter(pca_sub[y_sub == 1, 0], pca_sub[y_sub == 1, 1], c="#DC2626", alpha=0.3, label="Phishing (Fondo)")
        
        # Punto actual
        coord_actual = modelo_pca.transform(vector_x)[0]
        ax.scatter(coord_actual[0], coord_actual[1], c="#F59E0B", s=250, marker="*", edgecolors="black", linewidths=1.5, label="URL Actual Inspeccionada", zorder=5)
        
        ax.set_title("Proyección en Componentes Principales (PC1 vs PC2)")
        ax.set_xlabel("Primer Componente Principal (18.27% varianza)")
        ax.set_ylabel("Segundo Componente Principal (10.16% varianza)")
        ax.legend(loc="upper right")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
