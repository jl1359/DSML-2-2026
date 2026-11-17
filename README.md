# 🛡️ Sistema de Detección y Clasificación de Phishing en URLs mediante Machine Learning

**Materia:** Machine Learning y Ciencia de los Datos (2026)  
**Docente:** Lic. Patricia Rodríguez Bilbao  
**Área Asignada:** Seguridad  

---

## 📌 Descripción del Proyecto
Este proyecto desarrolla un prototipo de software de ciberseguridad para la detección y mitigación de enlaces fraudulentos (*Phishing*). Se implementa una solución híbrida compuesta por:

1. **Modelos de Aprendizaje Profundo (Supervisados):** Redes neuronales (CNN-1D, LSTM, GRU) que analizan las URLs carácter por carácter para clasificar amenazas conocidas con alta precisión.
2. **Modelos de Aprendizaje No Supervisado:** Algoritmos de agrupamiento y detección de anomalías (DBSCAN, k-Means, Clustering Jerárquico) para identificar patrones atípicos y ataques no catalogados (*Zero-Day*).

---

## 📁 Estructura del Repositorio

```text
DSML-2-2026/
├── .gitignore                    # Reglas para no subir datos pesados ni temporales
├── README.md                     # Documentación principal del repositorio
├── requirements.txt              # Dependencias oficiales de Python
│
├── data/
│   └── phiusiil/                 # Dataset oficial del proyecto
│       ├── raw/                  # Dataset original (PhiUSIIL ZIP comprimido de 16 MB)
│       ├── processed/            # Datos limpios y estandarizados (train.csv / test.csv)
│       └── samples/              # Muestra ligera (30k filas) para desarrollo ágil
│
├── notebooks/                    # Cuadernos Jupyter por integrante y modelo
│   ├── 01_preprocesamiento/      # Cuaderno de EDA, limpieza y Z-Score
│   ├── 02_aprendizaje_profundo/  # Cuadernos de CNN-1D, LSTM y GRU
│   └── 03_aprendizaje_no_supervisado/ # Cuadernos de DBSCAN, k-Means y Jerárquico
│
├── src/                          # Código fuente modular en Python
│   ├── __init__.py
│   ├── data_loader.py            # Carga y descompresión automática de datos
│   ├── preprocessing.py          # Pipeline de limpieza y estandarización Z-Score
│   └── models/                   # Módulos de definición de los modelos
│
├── saved_models/                 # Modelos entrenados y serializados (.keras, .pkl)
│   └── phiusiil/
│
└── app/                          # Prototipo interactivo en Streamlit
    └── main.py                   # Dashboard web de visualización y pruebas en vivo
```

---

## 🚀 Guía de Instalación y Preparación del Entorno

### 1. Clonar el repositorio
```bash
git clone https://github.com/jl1359/DSML-2-2026.git
cd DSML-2-2026
```

### 2. Crear y activar el entorno virtual

**Opción con `uv` (Recomendada - Ultrarrápida):**
```bash
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

**Opción con `python` estándar:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 ¿Cómo cargar los datos en tu código o notebook?

No necesitas descomprimir archivos manualmente. Desde cualquier notebook o script de Python:

```python
from src.data_loader import load_phiusiil_processed

# Carga automática de train (80% = 188k filas) y test (20% = 47k filas) listos y estandarizados
train_df, test_df = load_phiusiil_processed()
```

---

## 💻 Ejecutar la Aplicación Interactiva (Streamlit)

```bash
streamlit run app/main.py
```
