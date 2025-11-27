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

A partir de aquí, se podrá ejecutar cualquiera de los experimentos definidos en los archivos `.toml` dentro de la carpeta `experiments`.

Para ejecutar un experimento concreto, se utilizará el siguiente comando:

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
