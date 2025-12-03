import os
import matplotlib.pyplot as plt
from task import load_data

num_partitions = 10
batch_size = 32
num_classes = 10

# CARPETA PARA GUARDAR EL HISTOGRAMA
os.makedirs("histogramas", exist_ok=True)

# Matriz de frecuencias: filas = clases, columnas = clientes
frecuencias = []

for pid in range(num_partitions):
    trainloader, _ = load_data(pid, num_partitions, batch_size)
    
    # CONTAR ETIQUETAS DE CADA CLIENTE
    labels = []
    for batch in trainloader:
        labels.extend(batch["label"].tolist())
    
    counts = [labels.count(c) for c in range(num_classes)]
    frecuencias.append(counts)

# TRANSPONER LA MATRIZ PARA TENER FILAS = CLASES, COLUMNAS = CLIENTES
frecuencias = list(zip(*frecuencias))

# CREAR GRÁFICO DE BARRAS APILADAS
plt.figure(figsize=(10, 6))
bottom = [0] * num_partitions
for clase_idx, counts in enumerate(frecuencias):
    plt.bar(range(num_partitions), counts, bottom=bottom, label=f"Clase {clase_idx}")
    bottom = [sum(x) for x in zip(bottom, counts)]

plt.xlabel("Clientes")
plt.ylabel("Frecuencia")
plt.title("Distribución de clases por cliente")
plt.xticks(range(num_partitions), [f"Cliente {i}" for i in range(num_partitions)])
plt.legend(title="Clases")
plt.tight_layout()

# GUARDAR GRÁFICO
plt.savefig("histogramas/histograma.png")
plt.close()
print("Gráfico de barras apiladas guardado en 'histogramas/histograma.png'")