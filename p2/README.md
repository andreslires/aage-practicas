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

2-AprendizajeContinuo/2-AprendizajeContinuo.ipynb

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
