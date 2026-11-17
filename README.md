# Sistema de Deteccion y Clasificacion de Phishing en URLs mediante Machine Learning

**Materia:** Machine Learning y Ciencia de los Datos (2026)  
**Docente:** Lic. Patricia Rodriguez Bilbao  
**Area Asignada:** Seguridad  

---

## 1. Descripcion del Proyecto
Este proyecto desarrolla un prototipo de software de ciberseguridad para la deteccion y clasificacion de URLs fraudulentas (Phishing). La solucion esta conformada por dos enfoques complementarios:

1. **Modelos de Aprendizaje Profundo (Supervisados):** Redes neuronales (MLP, CNN-1D, LSTM/GRU) disenadas para la clasificacion precisa de enlaces mediante el analisis de caracteristicas lexico-estructurales.
2. **Modelos de Aprendizaje No Supervisado:** Tecnicas de agrupamiento y deteccion de anomalias (K-Means, DBSCAN, Clustering Jerarquico) orientadas a la identificacion de patrones no catalogados y ataques emergentes.

---

## 2. Estructura del Repositorio

```text
DSML-2-2026/
├── .gitignore                    # Reglas de exclusion de archivos pesados y temporales
├── README.md                     # Documentacion principal del repositorio
├── requirements.txt              # Dependencias oficiales de Python
│
├── data/
│   └── phiusiil/                 # Dataset oficial del proyecto
│       ├── raw/                  # Dataset original (PhiUSIIL ZIP comprimido)
│       ├── processed/            # Datos limpios y normalizados (train.csv / test.csv)
│       └── samples/              # Muestra balanceada de prueba rapida (30,000 registros)
│
├── notebooks/                    # Cuadernos Jupyter ordenados por etapa y modelo
│   ├── 01_preprocesamiento/      # Analisis Exploratorio (EDA), limpieza y escalado
│   ├── 02_deep_learning/         # Experimentacion de Redes Neuronales
│   └── 03_no_supervisado/        # Experimentacion de Clustering y Anomalias
│
├── models/                       # Scripts y artefactos modulares por familia de algoritmos
│   ├── deep_learning/
│   │   ├── guardados/            # Modelos entrenados y serializados (.keras, .h5)
│   │   └── resultados/           # Curvas de perdida/precision, matrices de confusion y ROC
│   └── unsupervised/
│       ├── guardados/            # Modelos serializados (.pkl, .joblib)
│       └── resultados/           # Graficas de metodo del codo, silueta y clusters PCA
│
├── src/                          # Codigo modular y funciones compartidas
│   ├── __init__.py
│   ├── data_loader.py            # Carga y descompresion automatica de datasets
│   ├── preprocessing.py          # Pipeline de limpieza y estandarizacion Z-Score
│   └── utils.py                  # Utilidades compartidas y calculo de metricas
│
└── app/                          # Prototipo de aplicacion web
    └── main.py                   # Dashboard en Streamlit para clasificacion y visualizacion
```

---

## 3. Instalacion y Preparacion del Entorno

### Clonar el repositorio
```bash
git clone https://github.com/jl1359/DSML-2-2026.git
cd DSML-2-2026
```

### Crear y activar el entorno virtual

**Con uv (Recomendado):**
```bash
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

**Con Python estandar:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Uso del Modulo de Datos

El modulo `src.data_loader` automatiza la descompresion y lectura de los datos:

```python
from src.data_loader import load_phiusiil_processed, load_phiusiil_sample

# Conjuntos completos para entrenamiento (188,260 registros) y prueba (47,065 registros):
train_df, test_df = load_phiusiil_processed()

# Muestra rapida balanceada para pruebas locales (30,000 registros):
sample_df = load_phiusiil_sample()
```

---

## 5. Ejecucion del Dashboard Interactivo

```bash
streamlit run app/main.py
```
