# Práctica 1: Predicción de Retrasos en Vuelos con PySpark

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
aage-practica1
│
├── p1_AndresLires_AngelVilarino.py           # Script principal
├── full_data_flightdelay.csv                 # Dataset original
├── p1_AndresLires_AngelVilarino.ipynb        # Notebook opcional
├── p1_Memoria_AndresLires_AngelVilarino.pdf  # Memoria del proyecto
└── README.md                                 # Este archivo
```
---

# Práctica 2: Aprendizaje Federado y Continuo

**Autores:** Andrés Lires Saborido, Ángel Vilariño García

**Curso:** 2025-2026

---

## Estructura de la práctica

El contenido de la práctica se organiza en la siguiente estructura de carpetas y archivos:

```
aage-practica2/
├── 1-AprendizajeFederado/
│   ├── server_app.py
│   ├── client_app.py
│   ├── task.py
│   ├── metrics.py
│   ├── histogramas.py
│   ├── pyproject.toml
│   ├── experiments/
│   │   ├── ...
│   ├── graficas/
│   │   ├── ...
│   ├── metrics/
│   │   ├── ...
│   └── histograms/
│   │   ├── ...
├── 2-AprendizajeContinuo/
│   ├── 2-AprendizajeContinuo.py
│   └── graficas/
├── README.md
└── memoria.md
```

Para más información sobre la implementación y experimentación, consulte el archivo `memoria.md`.

## Intrucciones de ejecución

### Parte I: Aprendizaje Federado con Flower

La ejecución de los experimentos de Aprendizaje Federado se harán por terminal.

Como se ha visto en clase, se utilizará el siguiente comando base:

```bash
flwr run .
```

En el caso de esta práctica será importante acceder a la carpeta `1-AprendizajeFederado` antes de ejecutar el comando. 
Para esto, si se abre la terminal en la carpeta raíz de la práctica, se deberá ejecutar el siguiente comando:

```bash
cd 1-AprendizajeFederado
```

Situados en este directorio, se podrá ejecutar por terminal cualquiera de los experimentos definidos. 

Para tratar de facilitar la experimentación, se ha intentado buscar una manera de que la selección del modelo a utilizar, el método de agregación y los hiperparámetros no implique la modificación de los scripts principales (`server_app.py`, `client_app.py` y `task.py`). Para ello, se han optimizado los códigos para que se lean estos parámetros desde diferentes archivos `.toml`. Se han incorporado condicionales y variables globales para que, en función del archivo `.toml` que se utilice al ejecutar el comando `flwr run .`, se seleccionen los parámetros deseados.

Para ejecutar un experimento concreto de los definidos en la carpeta `experiments`, se utilizará el siguiente comando:

```bash
flwr run . --run-config experiments/<nombre_experimento>.toml
```

Con esto, se entrenará el modelo elegido con el método de agregación y los hiperparámetros definidos en el archivo `.toml`.

Las gráficas comparativas de las métricas obtenidas en las distintas experimentaciones se han obtenido ejecutando el archivo `metrics.py` de la siguiente manera:

```bash
python metrics.py
```

De igual manera, los histogramas de distribución de clases entre clientes se han generado ejecutando el archivo `histogramas.py`:

```bash
python histogramas.py
```

### Parte II: Aprendizaje Continuo con River

La segunda parte de la práctica se desarrolla en el notebook:

`2-AprendizajeContinuo/2-AprendizajeContinuo.ipynb`

Este archivo contiene toda la experimentación relacionada con aprendizaje en *streaming*, detección de *concept drift* y modelos adaptativos. 
Las dependencias para ejecutar el notebook se pueden instalar de dos formas en un entorno virtual:
1. Usando pip:

```bash
pip install river scikit-learn matplotlib
```

2. Usando conda:

```bash
conda install -c conda-forge river scikit-learn matplotlib
```

Una vez instaladas, ejecutar el notebook es tan sencillo como abrirlo con *VS Code*, *Jupyter Notebook* o *Jupyter Lab* y ejecutar las celdas en orden. 

Por último, cabe destacar que las gráficas generadas durante la experimentación, para la comparación de modelos de aprendizaje continuo, serán guardadas en la carpeta `2-AprendizajeContinuo/graficas`.
