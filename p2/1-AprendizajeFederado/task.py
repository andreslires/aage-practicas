import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt 
import os


# Modelo base: MLP Simple
class MLPSimple(nn.Module):
    """Simple MLP model for Fashion-MNIST."""
    def __init__(self):
        super(MLPSimple, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

fds = None  # Cache FederatedDataset

# Normalización correcta para Fashion-MNIST
# Estas son las estadísticas estándar del dataset Fashion-MNIST
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3205
pytorch_transforms = Compose([ToTensor(), Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

# Funcion para graficar histogramas
def plot_histograma_cliente(labels, client_id):
    """Guarda el histograma de clases para un cliente."""  
    # Crear carpeta si no existe
    os.makedirs("histogramas", exist_ok=True)

    plt.figure(figsize=(5, 3))
    plt.hist(labels, bins=range(11), align='left', rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de clases - Cliente {client_id}")
    plt.tight_layout()

    # Guardar imagen
    filename = f"histogramas/cliente_{client_id}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Histograma guardado para el Cliente {client_id}")

# Función para guardar todos los histogramas de clientes
def plot_all_histograms(num_partitions):
    """Plot histograms for all clients."""
    global fds
    partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=0.1)
    fds = FederatedDataset(
        dataset="fashion_mnist",
        partitioners={"train": partitioner},
    )

    for pid in range(num_partitions):
        part = fds.load_partition(pid)
        part.set_format("numpy")   
        labels = [int(l) for l in part["label"]]
        plot_histograma_cliente(labels, pid)

# Función para graficar curvas de entrenamiento
def plot_training_curves(metrics_history, save_path_prefix="metrics"):
    """Plot training curves for loss and accuracy."""
    os.makedirs("graficas", exist_ok=True)

    rounds = [m["round"] for m in metrics_history]
    losses = [m["loss"] for m in metrics_history]
    accuracies = [m["accuracy"] for m in metrics_history]

    plt.figure()
    plt.plot(rounds, losses, marker="o")
    plt.xlabel("Ronda")
    plt.ylabel("Pérdida")
    plt.title("Pérdida agregada por ronda")
    plt.grid(True)
    plt.savefig(f"graficas/{save_path_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(rounds, accuracies, marker="o")
    plt.xlabel("Ronda")
    plt.ylabel("Precisión")
    plt.title("Precisión agregada por ronda")
    plt.grid(True)
    plt.savefig(f"graficas/{save_path_prefix}_accuracy.png")
    plt.close()


# Cargamos los datos y particionamos con Dritchlet alpha<=0.1
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition Fashion-MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Particionado basado en distribución Dirichlet con alpha<=0.1
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=0.1)
        fds = FederatedDataset(
            dataset="fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition.set_format("torch")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, shuffle=False
    )
    return trainloader, testloader

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("fashion_mnist", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
