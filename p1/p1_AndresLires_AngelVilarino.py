# Importamos las librerías necesarias

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, sum, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, RobustScaler, UnivariateFeatureSelector, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC, MultilayerPerceptronClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Configuración de Spark con la ruta correcta

# Ruta Andrés
# os.environ['JAVA_HOME'] = 'C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot'
# Ruta Vila
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home'
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

# Carga de datos y mostrado de las primeras filas

df = spark.read.csv('full_data_flightdelay.csv', header=True, inferSchema=True)
df.show(5)


## ANALISIS EXPLORATORIO DE DATOS (EDA) ##


# Mostramos tipos de datos de cada columna (schema)

df.printSchema()

# Mostramos número de filas y columnas

print('Número de filas:', df.count())
print('Número de columnas:', len(df.columns))

# Mostramos la distribución de la variable objetivo (DEP_DEL15)

df.groupBy('DEP_DEL15').count().show()

# Existencia de valores nulos en el dataset

null_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
null_counts.show()

# Análisis estadístico descriptivo

df.summary().show()

# Contamos cuantos valores diferentes hay en cada columna categórica
categorical_columns = ["DEP_TIME_BLK", "CARRIER_NAME", "DEPARTING_AIRPORT", "PREVIOUS_AIRPORT"]

for column in categorical_columns:
    distinct_count = df.select(column).distinct().count()
    print(f"Columna '{column}' tiene {distinct_count} valores distintos.")

# Distribución de variables categóricas

for column in categorical_columns:
    print(f"Distribución de la columna '{column}':")
    df.groupBy(column).count().orderBy('count', ascending=False).show(5)  # Mostramos los 10 más comunes

# Distribución de la variable objetivo respecto a otras variables (numéricas y categóricas)

# Comparamos el número de retrasos y no retrasos por mes

df_grouped = (df.groupBy("MONTH", "DEP_DEL15").count().orderBy("MONTH"))

df_pandas = df_grouped.toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_pandas, x="MONTH", y="count", hue="DEP_DEL15", palette=["skyblue", "salmon"])
plt.title("Número de retrasos y no retrasos por mes")
plt.xlabel("Mes")
plt.ylabel("Número de vuelos")
plt.legend(title="Retraso", labels=["No", "Sí"], labelcolor=["skyblue", "salmon"])
plt.show()

# Comparamos los vuelos con y sin retraso en cada aerolínea

df_grouped_airline = (df.groupBy("CARRIER_NAME", "DEP_DEL15").count().orderBy("CARRIER_NAME"))
df_pandas_airline = df_grouped_airline.toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_pandas_airline, y="CARRIER_NAME", x="count", hue="DEP_DEL15", palette=["skyblue", "salmon"])
plt.title("Número de retrasos y no retrasos por aerolínea")
plt.xlabel("Aerolínea")
plt.ylabel("Número de vuelos")
plt.legend(title="Retraso", labels=["No", "Sí"], labelcolor=["skyblue", "salmon"])
plt.xticks(rotation=45)
plt.show()

#  Análisis entre los retrasos y no con las variables relacionadas con la climatología (PRCP, SNOW, SNWD, TMAX, AWND)

df_weather = df.select("PRCP", "SNOW", "SNWD", "TMAX", "AWND", "DEP_DEL15")
df_weather_pandas = df_weather.toPandas()

plt.figure(figsize=(15, 10))
for i, column in enumerate(df_weather_pandas.columns[:-1], 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=df_weather_pandas, x="DEP_DEL15", y=column, palette=["skyblue", "salmon"], hue="DEP_DEL15", legend=False)
    plt.title(f"Distribución de {column} por retraso")
    plt.xlabel("Retraso")
    plt.ylabel(column)
plt.tight_layout()
plt.show()

# Tabla con la media de las variables meteorológicas agrupadas por retraso o no retraso con el df de spark

df_weather_grouped = df.groupBy("DEP_DEL15").mean("PRCP", "SNOW", "SNWD", "TMAX", "AWND")
df_weather_grouped.show()

# Ahora con el resto de variables numéricas del dataset

numerical_columns = ["DISTANCE_GROUP", "SEGMENT_NUMBER", "CONCURRENT_FLIGHTS", "NUMBER_OF_SEATS",
                     "AIRPORT_FLIGHTS_MONTH", "AIRLINE_FLIGHTS_MONTH", "AIRLINE_AIRPORT_FLIGHTS_MONTH", "AVG_MONTHLY_PASS_AIRPORT",
                     "AVG_MONTHLY_PASS_AIRLINE", "FLT_ATTENDANTS_PER_PASS", "GROUND_SERV_PER_PASS"]

# Usamos el desempaquetado * para pasar la lista como argumentos posicionales.
df_numerical_grouped = df.groupBy("DEP_DEL15").mean(*numerical_columns)
df_numerical_grouped.show()


# Reducimos el número de filas del dataset, manteniendo la proporción de vuelos con y sin retraso, usando stratified sampling

fractions = {0: 0.1, 1: 0.1}
df_sampled = df.stat.sampleBy("DEP_DEL15", fractions, seed=42)
print('Número de filas después del muestreo estratificado:', df_sampled.count())

print("Número de vuelos con y sin retraso después del muestreo estratificado:")
print("Con retraso:", df_sampled.filter(df_sampled.DEP_DEL15 == 1).count())
print("Sin retraso:", df_sampled.filter(df_sampled.DEP_DEL15 == 0).count())

# Calculamos el peso de cada clase para manejar el desbalanceo

# Calcular número de ejemplos por clase
class_counts = df_sampled.groupBy("DEP_DEL15").count().collect()
counts = {row["DEP_DEL15"]: row["count"] for row in class_counts}
majority = max(counts.values())
weights = {label: majority / count for label, count in counts.items()}

print("Pesos aplicados a cada clase:", weights)

# Añadir columna de pesos
df_weighted = df_sampled.withColumn("classWeightCol", when(col("DEP_DEL15") == 1, weights[1]).otherwise(weights[0]))

# División en train y test

train_data, test_data = df_weighted.randomSplit([0.8, 0.2], seed=42)

print(f'Tamaño del conjunto de entrenamiento: {train_data.count()}')
print(f'Tamaño del conjunto de test: {test_data.count()}')

# Creación del pipeline de preprocesamiento

# Columnas categóricas y numéricas
categorical_columns = ["DEP_TIME_BLK", "CARRIER_NAME", "DEPARTING_AIRPORT"]
numerical_columns = ["MONTH", "DAY_OF_WEEK", "DISTANCE_GROUP", "SEGMENT_NUMBER",
                     "CONCURRENT_FLIGHTS", "NUMBER_OF_SEATS", "AIRPORT_FLIGHTS_MONTH",
                     "AIRLINE_FLIGHTS_MONTH", "AIRLINE_AIRPORT_FLIGHTS_MONTH", "AVG_MONTHLY_PASS_AIRPORT",
                     "AVG_MONTHLY_PASS_AIRLINE", "FLT_ATTENDANTS_PER_PASS", "GROUND_SERV_PER_PASS",
                     "PLANE_AGE", "LATITUDE", "LONGITUDE", "PRCP", "SNOW", "SNWD", "TMAX", "AWND"]

# Index y OHE
indexers = StringIndexer(inputCols=categorical_columns, outputCols=[c+"_index" for c in categorical_columns])
encoder = OneHotEncoder(inputCols=indexers.getOutputCols(), outputCols=[c+"_ohe" for c in categorical_columns])

# Ensamblador y escalador
vecAssembler = VectorAssembler(inputCols=encoder.getOutputCols() + numerical_columns, outputCol="features")
scaler = RobustScaler(inputCol="features", outputCol="finalFeatures", withCentering=True, withScaling=True)

pipeline_weighted = Pipeline(stages=[indexers, encoder, vecAssembler, scaler])

pipeline_weighted_model = pipeline_weighted.fit(train_data)
train_weighted_scaled = pipeline_weighted_model.transform(train_data)
test_weighted_scaled = pipeline_weighted_model.transform(test_data)
train_weighted_scaled.select("finalFeatures", "DEP_DEL15", "classWeightCol").show(5)


# Entrenamiento con un modelo MultilayerPerceptronClassifier usando los pesos de clase

# El MLP usará 'finalFeatures' (producido por pipeline_weighted)
input_size = train_weighted_scaled.select("finalFeatures").first()[0].size

model = MultilayerPerceptronClassifier(featuresCol='finalFeatures', labelCol='DEP_DEL15', layers=[input_size, 10, 5, 2], maxIter=100, seed=42)

evaluator = MulticlassClassificationEvaluator(labelCol='DEP_DEL15', predictionCol='prediction')

model_fitted = model.fit(train_weighted_scaled)
predictions = model_fitted.transform(test_weighted_scaled)

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "PrecisionByLabel"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "RecallByLabel"})

print(f'Accuracy del modelo MLP: {accuracy:.4f}')
print(f'F1 del modelo MLP: {f1:.4f}')
print(f'Precisión del modelo MLP: {precision:.4f}')
print(f'Recall del modelo MLP: {recall:.4f}')

