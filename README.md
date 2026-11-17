# 🛡️ Sistema de Detección y Clasificación de Phishing en URLs mediante Machine Learning

**Materia:** Machine Learning y Ciencia de los Datos (2026)  
**Docente:** Lic. Patricia Rodríguez Bilbao  
**Área Asignada:** Seguridad  

---

## 📌 Descripción del Proyecto
Este proyecto implementa una solución integral para mitigar el fraude cibernético y el robo de credenciales mediante el análisis avanzado de URLs. Se resuelve el problema combinando:
1. **Modelos de Aprendizaje Profundo (Supervisados):** Redes neuronales convolucionales y recurrentes (CNN-1D, LSTM, GRU) para clasificar enlaces como legítimos o fraudulentos a partir de secuencias de caracteres.
2. **Modelos de Aprendizaje No Supervisado:** Algoritmos de agrupamiento y detección de anomalías (DBSCAN, k-Means, Clustering Jerárquico) para aislar patrones atípicos y detectar amenazas de día cero (*Zero-Day*).

---

## 👥 Asignación de Roles (Equipo de 8 Integrantes)

| # | Integrante | Rol Principal | Componente Asignado |
| :-: | :--- | :--- | :--- |
| **1** | Integrante 1 | Ingeniería de Datos | Carga, limpieza, balanceo (SMOTE/submuestreo) y EDA del dataset PhiUSIIL. |
| **2** | Integrante 2 | Deep Learning 1 | Red Convolucional 1D (**CNN-1D**) con capas de pooling y filtros locales. |
| **3** | Integrante 3 | Deep Learning 2 | Red Recurrente **LSTM** con celdas de memoria y compuertas anti-desvanecimiento. |
| **4** | Integrante 4 | Deep Learning 3 | Red Recurrente **GRU** optimizada para eficiencia en secuencias. |
| **5** | Integrante 5 | No Supervisado 1 | **k-Means** (MacQueen 1967) y análisis de distancias intra/inter cluster. |
| **6** | Integrante 6 | No Supervisado 2 | **DBSCAN** (Ester et al. 1996) para detección de densidad y ruido/outliers. |
| **7** | Integrante 7 | No Supervisado 3 | **Clustering Jerárquico Aglomerativo** y análisis de dendrogramas. |
| **8** | Integrante 8 | Software y Dashboard | Prototipo web interactivo en **Streamlit** y coordinación de informes. |

---

## 📁 Estructura del Repositorio

```text
DSML-2-2026/
├── .gitignore                    # Reglas para no subir datos pesados ni caches
├── README.md                     # Documentación principal del repositorio
├── requirements.txt              # Dependencias oficiales de Python
│
├── data/                         # Almacenamiento de datasets (ignorado por Git)
│   ├── raw/                      # Dataset original (ej. PhiUSIIL CSV)
│   ├── processed/                # Datasets limpios y particionados (train/test)
│   └── samples/                  # Muestras ligeras para desarrollo rápido
│
├── notebooks/                    # Cuadernos Jupyter por integrante/modelo
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_deep_learning_cnn.ipynb
│   ├── 03_deep_learning_lstm.ipynb
│   ├── 04_deep_learning_gru.ipynb
│   ├── 05_unsupervised_kmeans.ipynb
│   ├── 06_unsupervised_dbscan.ipynb
│   └── 07_unsupervised_hierarchical.ipynb
│
├── src/                          # Código modular y reutilizable
│   ├── __init__.py
│   ├── data_loader.py            # Funciones para leer y procesar datos
│   ├── feature_extraction.py     # Extracción de características léxicas
│   ├── utils.py                  # Métricas (ROC, F1, Silueta) y gráficas
│   └── models/
│       ├── deep_learning/        # Scripts de CNN, LSTM y GRU
│       └── unsupervised/         # Scripts de k-Means, DBSCAN y Jerárquico
│
├── saved_models/                 # Modelos serializados (.keras, .pkl)
├── app/                          # Prototipo en Streamlit
│   └── main.py                   # Dashboard web de visualización y pruebas
└── docs/                         # Informes académicos según el formato de la docente
    ├── pre_informe_1/
    ├── pre_informe_2/
    └── informe_final/
```

---

## 🚀 Guía de Instalación y Preparación del Entorno

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/DSML-2-2026.git
cd DSML-2-2026
```

### 2. Crear y activar el entorno virtual

**Con `uv` (Recomendado - Ultrarrápido):**
```bash
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

**Con `venv` estándar de Python:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación interactiva (Streamlit)
```bash
streamlit run app/main.py
```
