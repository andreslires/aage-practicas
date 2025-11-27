import matplotlib.pyplot as plt
import pandas as pd
import os

# FUNCIÓN PRINCIPAL PARA HACER GRÁFICAS
def plot_metrics(dataframes: list, metric: str, title: str, xlabel: str, ylabel: str, labels: list, name_file: str = None):
    plt.figure(figsize=(10, 6))
    
    for i, df in enumerate(dataframes):
        plt.plot(df[metric], label=labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Crear carpeta si no existe
    os.makedirs("graficas", exist_ok=True)
    plt.savefig(f"graficas/{name_file}.png")
    plt.close()


if __name__ == "__main__":

    # LEER CSVs
    metrics_FedAvg_MLPSimple = pd.read_csv("metrics\\metrics_FedAvg_MLPSimple.csv")
    metrics_FedProx_MLPSimple = pd.read_csv("metrics\\metrics_FedProx_MLPSimple.csv")
    metrics_FedAvg_CNNModel = pd.read_csv("metrics\\metrics_FedAvg_CNNModel.csv")
    metrics_FedProx_CNNModel = pd.read_csv("metrics\\metrics_FedProx_CNNModel.csv")

    # MLPSimple: FedAvg vs FedProx
    plot_metrics(
        dataframes=[metrics_FedAvg_MLPSimple, metrics_FedProx_MLPSimple],
        metric="accuracy",
        title="Comparación de Accuracy: FedAvg vs FedProx (MLPSimple)",
        xlabel="Rondas",
        ylabel="Accuracy",
        labels=["FedAvg", "FedProx"],
        name_file="Accuracy_FedAvg_vs_FedProx_MLPSimple"
    )
    plot_metrics(
        dataframes=[metrics_FedAvg_MLPSimple, metrics_FedProx_MLPSimple],
        metric="loss",
        title="Comparación de Loss: FedAvg vs FedProx (MLPSimple)",
        xlabel="Rondas",
        ylabel="Loss",
        labels=["FedAvg", "FedProx"],
        name_file="Loss_FedAvg_vs_FedProx_MLPSimple"
    )

    # CNNModel: FedAvg vs FedProx
    plot_metrics(
        dataframes=[metrics_FedAvg_CNNModel, metrics_FedProx_CNNModel],
        metric="accuracy",
        title="Comparación de Accuracy: FedAvg vs FedProx (CNNModel)",
        xlabel="Rondas",
        ylabel="Accuracy",
        labels=["FedAvg", "FedProx"],
        name_file="Accuracy_FedAvg_vs_FedProx_CNNModel"
    )
    plot_metrics(
        dataframes=[metrics_FedAvg_CNNModel, metrics_FedProx_CNNModel],
        metric="loss",
        title="Comparación de Loss: FedAvg vs FedProx (CNNModel)",
        xlabel="Rondas",
        ylabel="Loss",
        labels=["FedAvg", "FedProx"],
        name_file="Loss_FedAvg_vs_FedProx_CNNModel"
    )

    # COMPARACIÓN GENERAL
    plot_metrics(
        dataframes=[
            metrics_FedAvg_MLPSimple, metrics_FedProx_MLPSimple,
            metrics_FedAvg_CNNModel, metrics_FedProx_CNNModel
        ],
        metric="accuracy",
        title="Comparación de Accuracy: FedAvg vs FedProx (MLPSimple vs CNNModel)",
        xlabel="Rondas",
        ylabel="Accuracy",
        labels=["FedAvg MLPSimple", "FedProx MLPSimple", "FedAvg CNNModel", "FedProx CNNModel"],
        name_file="Accuracy_Comparacion_General"
    )
    plot_metrics(
        dataframes=[
            metrics_FedAvg_MLPSimple, metrics_FedProx_MLPSimple,
            metrics_FedAvg_CNNModel, metrics_FedProx_CNNModel
        ],
        metric="loss",
        title="Comparación de Loss: FedAvg vs FedProx (MLPSimple vs CNNModel)",
        xlabel="Rondas",
        ylabel="Loss",
        labels=["FedAvg MLPSimple", "FedProx MLPSimple", "FedAvg CNNModel", "FedProx CNNModel"],
        name_file="Loss_Comparacion_General"
    )
