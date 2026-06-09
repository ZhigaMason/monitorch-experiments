import argparse
import torch
import torch.nn as nn
import yaml
import urllib.request
import os
import pandas as pd
import numpy as np
from optimizer_utils import get_optimizer

from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import RecorderVisualizer
from monitorch import lens


# --- Data Preparation ---
def get_dataloader(seq_len, batch_size):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    if not os.path.exists("daily-min-temperatures.csv"):
        print("Downloading Daily Min Temperatures dataset...")
        urllib.request.urlretrieve(url, "daily-min-temperatures.csv")

    df = pd.read_csv("daily-min-temperatures.csv", parse_dates=["Date"])

    # Standardize the target variable
    temps = df["Temp"].values.astype(np.float32)
    temps_mean, temps_std = np.mean(temps), np.std(temps)
    temps = (temps - temps_mean) / temps_std

    # Create rolling windows
    X, Y = [], []
    for i in range(len(temps) - seq_len):
        X.append(temps[i : i + seq_len])
        Y.append(temps[i + seq_len])

    # Shape for PyTorch LSTM: (Batch, Seq_Len, Features)
    X = torch.tensor(np.array(X)).unsqueeze(-1)
    Y = torch.tensor(np.array(Y)).unsqueeze(-1)

    dataset = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# --- LSTM Architecture ---
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # PyTorch packs LSTM weights into 2D matrices, so Muon will capture and orthogonalize them
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Labeled 'fc' so it routes to AdamW
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Predict purely off the final hidden state of the sequence
        return self.fc(out[:, -1, :])


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
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(config["seq_len"], config["batch_size"])

    model = TimeSeriesLSTM(config["input_size"], config["hidden_size"]).to(device)
    optimizer = get_optimizer(
        model, config["optimizer"], config["lr"], config["weight_decay"]
    )
    criterion = nn.MSELoss()

    log_file = args.config_file.split("/")[-1].split(".")[0] + f"_{config['seed']}.pkl"
    inspector = PyTorchInspector(
        lenses=[
            lens.LossMetrics(loss_fn=criterion),
            lens.OutputActivation(),
            lens.ParameterGradientActivation(
                parameters=["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]
            ),
        ],
        module=model,
        visualizer=RecorderVisualizer(log_file),
    )

    model.train()
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        inspector.tick()
        print(
            f"Epoch {epoch + 1}/{config['epochs']} | Avg MSE Loss: {running_loss / len(dataloader):.4f}"
        )


if __name__ == "__main__":
    main()
