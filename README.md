# Sistema de Detección y Clasificación de Phishing en URLs mediante Machine Learning

**Materia:** Machine Learning y Ciencia de los Datos (2026)  
**Docente:** Lic. Patricia Rodríguez Bilbao  
**Área Asignada:** Seguridad (Ciberseguridad)  
**Dataset Oficial:** PhiUSIIL Phishing URL Dataset (UCI Benchmark 2024)  

---

## 1. Descripción del Proyecto

Este proyecto implementa una plataforma de ciberseguridad para la detección, clasificación y mitigación de URLs fraudulentas (*phishing*). La solución combina tres paradigmas de Machine Learning:

1. **Aprendizaje Profundo (Supervisado):** Redes neuronales profundas (**MLP**, **CNN-1D** y **LSTM**) diseñadas para clasificar enlaces maliciosos con una sensibilidad máxima (cero falsos negativos) en un conjunto de prueba independiente de 47,074 URLs.
2. **Aprendizaje No Supervisado:** Agrupamiento particional (**K-Means**) y detección de anomalías espaciales (**DBSCAN**) con reducción dimensional (**PCA**) para descubrir patrones no etiquetados y amenazas desconocidas (*Zero-Day*).
3. **Aprendizaje por Refuerzo:** Agente adaptativo (**Q-Learning**) para la toma de decisiones dinámicas de mitigación (Permitir, Bloquear, Aislar en Sandbox) minimizando el riesgo operativo.

---

## 2. Estructura del Repositorio

```text
DSML-2-2026/
├── .gitignore                    # Reglas de exclusión de datos pesados y artefactos
├── README.md                     # Documentación principal y guía de uso
├── requirements.txt              # Dependencias oficiales de Python
│
├── data/
│   └── phiusiil/                 # Dataset oficial del proyecto
│       ├── raw/                  # Dataset original comprimido (PhiUSIIL ZIP)
│       ├── processed/            # Conjuntos procesados (train.csv / test.csv)
│       └── samples/              # Muestra balanceada para pruebas rápidas
│
├── notebooks/                    # Cuadernos Jupyter por etapa y paradigma
│   ├── 01_preprocesamiento/      # Análisis Exploratorio (EDA), auditoría y Z-Score
│   ├── 02_deep_learning/         # Redes Neuronales (MLP, CNN-1D, LSTM)
│   ├── 03_no_supervisado/        # Clustering (K-Means, DBSCAN) y PCA
│   └── 04_refuerzo/              # Agente Q-Learning para mitigación dinámica
│
├── models/                       # Scripts modulares de entrenamiento y evaluación
│   ├── deep_learning/            # Entrenamiento de MLP, CNN-1D, LSTM y comparativa
│   ├── unsupervised/             # Scripts de K-Means, DBSCAN, PCA y métricas
│   └── reinforcement_learning/   # Agente Q-Learning y simulación de tráfico
│
├── src/                          # Código modular y funciones compartidas
│   ├── __init__.py
│   ├── data_loader.py            # Carga y descompresión automática del dataset
│   ├── preprocessing.py          # Pipeline de limpieza y estandarización Z-Score
│   └── interfaz.py               # Interfaz interactiva de ciberseguridad en Streamlit
│
└── app/                          # Acceso a la aplicación web
    └── main.py                   # Punto de entrada alternativo
```

---

## 3. Instalación y Preparación del Entorno

### 3.1 Clonar el repositorio
```bash
git clone https://github.com/jl1359/DSML-2-2026.git
cd DSML-2-2026
```

### 3.2 Crear y activar el entorno virtual (Python 3.11)

**Opción recomendada (con uv):**
```bash
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

**Opción estándar (con venv de Python):**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Instrucciones de Ejecución

Todos los scripts pueden ejecutarse de manera independiente desde la raíz del proyecto con el entorno virtual activado:

### 4.1 Modelos de Aprendizaje Profundo (Deep Learning)

Entrenamiento, validación y exportación de métricas y gráficos de cada arquitectura neuronal:

```bash
# 1. Entrenar y evaluar el Perceptrón Multicapa (MLP):
python models/deep_learning/entrenamiento_mlp.py

# 2. Entrenar y evaluar la Red Convolucional 1D (CNN-1D):
python models/deep_learning/entrenamiento_cnn.py

# 3. Entrenar y evaluar la Red Recurrente con Memoria (LSTM):
python models/deep_learning/entrenamiento_lstm.py

# 4. Generar tabla comparativa consolidada y gráfico comparativo de métricas:
python models/deep_learning/comparar_modelos_dl.py
```

### 4.2 Modelos de Aprendizaje No Supervisado (Clustering y Anomalías)

Ejecución de las técnicas de agrupamiento geométrico y detección de ataques Zero-Day:

```bash
# 1. Agrupamiento particional K-Means (método del codo y silueta):
python models/unsupervised/clustering_kmeans.py

# 2. Detección de anomalías espaciales con DBSCAN:
python models/unsupervised/clustering_dbscan.py

# 3. Reducción de dimensionalidad y varianza explicada con PCA (2D):
python models/unsupervised/analisis_pca.py

# 4. Evaluación comparativa de clustering (Pureza, Silueta, ARI, Homogeneidad):
python models/unsupervised/evaluar_clustering.py
```

### 4.3 Aprendizaje por Refuerzo (Q-Learning)

Entrenamiento del agente adaptativo y convergencia de recompensas acumuladas:

```bash
python models/reinforcement_learning/agente_qlearning.py
```

### 4.4 Cuadernos Interactivos (Jupyter Notebooks)

Para explorar interactivamente los cuadernos con código, gráficos incrustados y explicaciones teóricas:

```bash
jupyter lab
# O bien:
jupyter notebook
```
Navegar en el navegador web a la carpeta `notebooks/` y abrir el cuaderno de interés:
* `notebooks/01_preprocesamiento/01_eda_limpieza_phiusiil.ipynb`
* `notebooks/02_deep_learning/01_redes_neuronales_deep_learning.ipynb`
* `notebooks/03_no_supervisado/01_clustering_anomalias.ipynb`
* `notebooks/04_refuerzo/01_aprendizaje_por_refuerzo_qlearning.ipynb`

### 4.5 Interfaz Interactiva de Usuario (Streamlit)

Lanzar la interfaz web interactiva para clasificar URLs en tiempo real y visualizar proyecciones espaciales:

```bash
streamlit run src/interfaz.py
```
La aplicación se abrirá automáticamente en `http://localhost:8501`.

---

## 5. Uso del Módulo de Carga de Datos en Python

El módulo `src.data_loader` se encarga de descomprimir automáticamente el dataset y cargar las particiones oficiales:

```python
from src.data_loader import load_phiusiil_processed, load_phiusiil_sample

# 1. Cargar las particiones estratificadas oficiales (188,296 train / 47,074 test):
train_df, test_df = load_phiusiil_processed()

# 2. Cargar una muestra balanceada para pruebas rápidas en local:
sample_df = load_phiusiil_sample()
```
