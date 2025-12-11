"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

# IMPORTAR MODELO Y FUNCIONES DE EVALUACIÓN
from task import MLPSimple, load_centralized_dataset, test, CNNModel, MobileNet

# Create ServerApp
app = ServerApp()

# VARIABLES GLOBALES
metrics_file = None
nombre_estrategia = None
model_name = None

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # LEER VARIABLES GLOBALES
    global metrics_file, nombre_estrategia, model_name

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # ESTRATEGIA DE ENTRENAMIENTO DEFINIDA EN pyproject.toml
    strategy_name: str = context.run_config["strategy"]

    # MODELO A UTILIZAR DEFINIDO EN pyproject.toml
    model_name = context.run_config["model"]

    if model_name == "CNNModel":
        print("Using model: CNNModel")
        global_model = CNNModel()
    elif model_name == "MLPSimple":
        print("Using model: MLPSimple")
        global_model = MLPSimple()
    elif model_name == "MobileNet":
        print("Using model: MobileNet")
        global_model = MobileNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    arrays = ArrayRecord(global_model.state_dict())

    # INICIALIZAR CSV PARA GUARDAR METRICAS
    os.makedirs("metrics", exist_ok=True)
    nombre_estrategia = strategy_name

    metrics_file = f"metrics/metrics_{nombre_estrategia}_{model_name}_fraction-train_{context.run_config['fraction-train']}.csv"
    with open(metrics_file, "w") as f:
        f.write("round,accuracy,loss\n")
    print(f"Metricas guardadas en {metrics_file}")
  
    # SELECCIÓN DE ESTRATEGIA SEGÚN SE PASE POR CONFIGURACIÓN
    if strategy_name == "FedAvg":
        print("Usando estrategia: FedAvg")
        strategy = FedAvg(
            fraction_evaluate=fraction_evaluate,
            fraction_train=fraction_train,
            min_available_nodes=context.run_config["min-available-clients"],
        )

    elif strategy_name == "FedProx":
        print("Usando estrategia: FedProx")
        strategy = FedProx(
            fraction_evaluate=fraction_evaluate,
            fraction_train=fraction_train,
            min_available_nodes=context.run_config["min-available-clients"],
            # LEER EL VALOR DE MU DEFINIDO EN pyproject.toml
            # INFO: https://arxiv.org/abs/1812.06127
            proximal_mu=context.run_config["mu"],
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # SELECCIÓN DE MODELO SEGÚN SE HAYA DEFINIDO EN pyproject.toml
    global model_name

    if model_name == "CNNModel":
        model = CNNModel()
    elif model_name == "MLPSimple":
        model = MLPSimple()
    elif model_name == "MobileNet":
        model = MobileNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cpu")  # Usar CPU para consistencia
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # GUARDAR METRICAS EN CSV
    global metrics_file, nombre_estrategia
    with open(metrics_file, "a") as f:
        f.write(f"{server_round},{test_acc},{test_loss}\n")

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
