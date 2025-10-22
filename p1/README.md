# Predicción de Retrasos en Vuelos con PySpark

**Autores:** Andrés Lires y Ángel Vilarino  
**Archivo principal:** `p1_AndresLires_AngelVilarino.py`

---

## Descripción del proyecto

Este proyecto entrena modelos de machine learning en **PySpark** para predecir si un vuelo sufrirá retraso (`DEP_DEL15 = 1`) o no (`DEP_DEL15 = 0`), usando datos históricos del año 2019 de vuelos en EE.UU.

Se realiza:
- Análisis exploratorio de datos (EDA)
- Limpieza y transformación de variables
- Creación de *Pipeline* de preprocesamiento
- Entrenamiento y evaluación del modelo final elegido: MLP (Multi-Layer Perceptron)
- Optimización de hiperparámetros con Cross-Validation

## Requisitos previos

Antes de ejecutar el script, asegúrate de tener instalados los siguientes paquetes y entornos:

### 1. Instalar **Java** (requerido por Spark)
- **Windows:**  
  Descargar [Adoptium JDK 17](https://adoptium.net/)
- **Mac/Linux:**  
  Instalar mediante Homebrew:
``` bash
  brew install openjdk@17
```

### 2. Instalar las librerías necesarias

Crea un entorno virtual (opcional pero recomendado) y ejecuta:

``` bash
conda create -n pyspark_env python=3.12
conda activate pyspark_env
pip install pyspark pandas matplotlib seaborn
```

### 3. Configurar variables de entorno

En el script ya están definidas las rutas a `JAVA_HOME`. Revisa y ajusta según tu sistema:

#### Ejemplo Mac
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home'

#### Ejemplo Windows
os.environ['JAVA_HOME'] = 'C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot'

---

## Cómo ejecutar el proyecto

### Paso previo: Descargar el [dataset](https://www.kaggle.com/code/ohadvilnai580/predicting-2019-flight-delays-knn-svm-analysis#Merging)

### Opción 1 — Desde terminal
1. Coloca el archivo `full_data_flightdelay.csv` en la misma carpeta que el script `.py`.
2. Abre una terminal en esa carpeta.
3. Ejecuta:
   python p1_AndresLires_AngelVilarino.py

Esto lanzará Spark, cargará el dataset y comenzará el entrenamiento del modelo final y su optimización de hiperparámetros.

### Opción 2 — Desde Jupyter Notebook
1. Abre Jupyter Lab o Notebook.
2. Crea un nuevo archivo `.ipynb`.
3. Copia y pega el contenido del script.
4. Ejecuta las celdas una a una para seguir el flujo de trabajo paso a paso.

---

## Resultados esperados

El script imprimirá las métricas de rendimiento del modelo final en consola de la manera siguiente:

| Modelo | Accuracy | Precisión | Recall | F1-score |
|--------|-----------|-----------|---------|-----------|
| MLP    | 0.85      | 0.80      | 0.75    | 0.77      |

*(Los valores mostrados son solo un ejemplo).*

---

## Estructura del proyecto

```
Proyecto_RetrasosVuelos
│
├── p1_AndresLires_AngelVilarino.py     # Script principal
├── full_data_flightdelay.csv            # Dataset original
├── resultados/                         # Carpeta sugerida para guardar métricas o modelos
└── README.md                           # Este archivo
```