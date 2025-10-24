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
os.environ['JAVA_HOME'] = 'C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot'
# Ruta Vila
# os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home'
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

# Carga de datos y mostrado de las primeras filas

df = spark.read.csv('p1/full_data_flightdelay.csv', header=True, inferSchema=True)
df.show(5)
df.printSchema()

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
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall "})

print(f'Accuracy del modelo MLP: {accuracy:.4f}')
print(f'F1 del modelo MLP: {f1:.4f}')
print(f'Precisión del modelo MLP: {precision:.4f}')
print(f'Recall del modelo MLP: {recall:.4f}')

# Búsqueda de hiperparámetros con Cross-Validation para el MLP

# Aseguramos que el pipeline genera la columna 'features'
train_transformed = pipeline_weighted_model.transform(train_data)
test_transformed = pipeline_weighted_model.transform(test_data)

# Obtenemos el número de características de entrada
input_size = train_transformed.select("finalFeatures").first()[0].size
print(f"Número de features de entrada: {input_size}")

# Definimos el clasificador MLP
mlp = MultilayerPerceptronClassifier(featuresCol='finalFeatures', labelCol='DEP_DEL15', seed=42)

# Definimos una pequeña rejilla de hiperparámetros
paramGrid = (ParamGridBuilder()
             .addGrid(mlp.layers, [
                 [input_size, 10, 2],
                 [input_size, 15, 5, 2]
             ])
             .addGrid(mlp.maxIter, [50, 100])
             .addGrid(mlp.stepSize, [0.03, 0.1])
             .build())

# Evaluador
evaluator = MulticlassClassificationEvaluator(labelCol='DEP_DEL15',predictionCol='prediction',metricName='f1')

# CrossValidator
crossval = CrossValidator(estimator=mlp, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3, parallelism=3)

# Entrenamos el modelo con cross-validation
cv_model = crossval.fit(train_transformed)

# Recuperamos el mejor modelo
best_model = cv_model.bestModel
print("=== Mejores hiperparámetros encontrados ===")
print(f"Layers: {best_model.layers}")
print(f"MaxIter: {best_model.getMaxIter()}")
print(f"StepSize: {best_model.getStepSize()}")

# Evaluamos el modelo en el conjunto de prueba
predictions = cv_model.transform(test_transformed)

metrics = {
    "accuracy": "accuracy",
    "precision": "weightedPrecision",
    "recall": "weightedRecall",
    "f1": "f1"
}

for name, metric in metrics.items():
    value = evaluator.evaluate(predictions, {evaluator.metricName: metric})
    print(f"{name.capitalize()} del modelo MLP con CV: {value:.4f}")