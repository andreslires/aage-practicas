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
    metrics_FedAvg_CNNModel_local_epochs_3 = pd.read_csv("metrics\\metrics_FedAvg_CNNModel_local-epochs_3.csv")
    metrics_FedAvg_CNNModel_local_epochs_5 = pd.read_csv("metrics\\metrics_FedAvg_CNNModel_local-epochs_5.csv")
    metrics_FedAvg_CNNModel_fraction_train_01 = pd.read_csv("metrics\\metrics_FedAvg_CNNModel_fraction-train_0.1.csv")
    metrics_FedAvg_CNNModel_fraction_train_09 = pd.read_csv("metrics\\metrics_FedAvg_CNNModel_fraction-train_0.9.csv")
    metrics_MobileNet = pd.read_csv("metrics\\metrics_FedAvg_MobileNet.csv")

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

# Comparar FedAvg CNNModel con 1, 3 y 5 épocas locales
    plot_metrics(
        dataframes=[metrics_FedAvg_CNNModel, metrics_FedAvg_CNNModel_local_epochs_3, metrics_FedAvg_CNNModel_local_epochs_5],
        metric="accuracy",
        title="Comparación de Accuracy: FedAvg CNNModel con 1, 3 y 5 épocas locales",
        xlabel="Rondas",
        ylabel="Accuracy",
        labels=["1 época local", "3 épocas locales", "5 épocas locales"],
        name_file="Accuracy_FedAvg_CNNModel_local_epochs"
    )
    plot_metrics(
        dataframes=[metrics_FedAvg_CNNModel, metrics_FedAvg_CNNModel_local_epochs_3, metrics_FedAvg_CNNModel_local_epochs_5],
        metric="loss",
        title="Comparación de Loss: FedAvg CNNModel con 1, 3 y 5 épocas locales",
        xlabel="Rondas",
        ylabel="Loss",
        labels=["1 época local", "3 épocas locales", "5 épocas locales"],
        name_file="Loss_FedAvg_CNNModel_local_epochs"
    )

# Comparar FedAvg CNNModel con fraction train 0.1, 0.5 y 0.9
    plot_metrics(
        dataframes=[metrics_FedAvg_CNNModel, metrics_FedAvg_CNNModel_fraction_train_01, metrics_FedAvg_CNNModel_fraction_train_09],
        metric="accuracy",
        title="Comparación de Accuracy: FedAvg CNNModel con fraction train 0.1, 0.5 y 0.9",
        xlabel="Rondas",
        ylabel="Accuracy",
        labels=["Fraction train 0.5", "Fraction train 0.1", "Fraction train 0.9"],
        name_file="Accuracy_FedAvg_CNNModel_fraction_train"
    )
    plot_metrics(
        dataframes=[metrics_FedAvg_CNNModel, metrics_FedAvg_CNNModel_fraction_train_01, metrics_FedAvg_CNNModel_fraction_train_09],
        metric="loss",
        title="Comparación de Loss: FedAvg CNNModel con fraction train 0.1, 0.5 y 0.9",
        xlabel="Rondas",
        ylabel="Loss",
        labels=["Fraction train 0.5", "Fraction train 0.1", "Fraction train 0.9"],
        name_file="Loss_FedAvg_CNNModel_fraction_train"
    )

    # Graficar  MobileNet
    plot_metrics(
        dataframes=[metrics_MobileNet],
        metric="accuracy",
        title="Accuracy: FedAvg MobileNet",
        xlabel="Rondas",
        ylabel="Accuracy",
        labels=["FedAvg MobileNet"],
        name_file="Accuracy_FedAvg_MobileNet"
    )

    plot_metrics(
        dataframes=[metrics_MobileNet],
        metric="loss",
        title="Loss: FedAvg MobileNet",
        xlabel="Rondas",
        ylabel="Loss",
        labels=["FedAvg MobileNet"],
        name_file="Loss_FedAvg_MobileNet"
    )