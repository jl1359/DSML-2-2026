"""
Aplicación Interactiva Streamlit - Detección y Clasificación de Phishing en URLs
Área: Seguridad
Materia: Machine Learning y Ciencia de los Datos (2026)
Docente: Lic. Patricia Rodríguez Bilbao
"""
import streamlit as st

st.set_page_config(
    page_title="Phishing Shield ML",
    layout="wide"
)

st.title("Sistema de Detección y Clasificación de Phishing en URLs")
st.markdown("""
Plataforma integral de ciberseguridad impulsada por **Aprendizaje Profundo** (Supervisado) y **Aprendizaje No Supervisado** (Clustering y Detección de Anomalías).
""")

# Barra lateral para configuración
st.sidebar.header("Configuración del Análisis")
dl_model = st.sidebar.selectbox(
    "Modelo de Deep Learning:",
    ["CNN-1D (Convolucional)", "LSTM (Memoria a Corto Plazo)", "GRU (Gated Recurrent Unit)"]
)

unsup_model = st.sidebar.selectbox(
    "Modelo No Supervisado:",
    ["DBSCAN (Basado en Densidad)", "k-Means (Particiones)", "Clustering Jerárquico"]
)

# Entrada de usuario
url_input = st.text_input("Ingrese la URL a analizar:", placeholder="https://ejemplo-banco-seguro.com/login")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Diagnóstico Deep Learning")
    if st.button("Analizar con Deep Learning"):
        if url_input:
            st.info(f"Analizando '{url_input}' con modelo {dl_model}...")
            # Aquí se conectará la inferencia del modelo
        else:
            st.warning("Por favor, ingrese una URL válida.")

with col2:
    st.subheader("Diagnóstico No Supervisado")
    if st.button("Evaluar Anomalía"):
        if url_input:
            st.info(f"Calculando densidad y distancia con {unsup_model}...")
            # Aquí se conectará la inferencia del modelo
        else:
            st.warning("Por favor, ingrese una URL válida.")
