"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from task import MLPSimple, load_centralized_dataset, test, plot_all_histograms, plot_training_curves

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Edit: Graficar histogramas antes del entrenamiento
    plot_all_histograms(10)

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = MLPSimple()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    # strategy = FedAvg(
    #     fraction_evaluate=fraction_evaluate,
    #     fraction_train=fraction_train,
    #     min_available_nodes=context.run_config["min-available-clients"],
    # )

    # Edit: Inicializar FedProx strategy
    strategy = FedProx(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
        min_available_nodes=context.run_config["min-available-clients"]
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Edit: Graficar curvas de entrenamiento después del entrenamiento
    plot_training_curves(metrics_history, "FedProx")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

# Edit: Inicializar historial de métricas
metrics_history = []

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    # Edit: Modelo MLPSimple
    model = MLPSimple()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cpu")  # Usar CPU para consistencia
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Edit: Guardar métricas en el historial
    metrics_history.append({"round": server_round, "loss": test_loss, "accuracy": test_acc})

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
