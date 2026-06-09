import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
import random
import numpy as np
from optimizer_utils import get_optimizer

from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import RecorderVisualizer
from monitorch import lens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# A simple modern CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Naming is important: 'fc' tells the router to assign this to AdamW when using Muon
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config-file",
        type=str,
        help="Config file.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {config['experiment']} on {device} with {config['optimizer']}")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )

    devset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    devloader = torch.utils.data.DataLoader(
        devset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model, config["optimizer"], config["lr"], config["weight_decay"]
    )

    log_file = args.config_file.split("/")[-1].split(".")[0] + f"_{config['seed']}.pkl"
    inspector = PyTorchInspector(
        lenses=[
            lens.LossMetrics(loss_fn=criterion),
            lens.OutputActivation(),
            lens.ParameterGradientActivation(),
        ],
        module=model,
        visualizer=RecorderVisualizer(log_file),
    )

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for inputs, labels in devloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        inspector.tick()
        print(
            f"Epoch {epoch + 1}/{config['epochs']} | dev Loss: {running_loss / len(trainloader):.4f} | dev Acc: {100.0 * correct / total:.2f}%"
        )


if __name__ == "__main__":
    main()
