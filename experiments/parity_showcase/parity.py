#!/usr/bin/env python3
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from monitorch.inspector import PyTorchInspector
from monitorch import lens
from monitorch.visualizer import RecorderVisualizer, PlayerVisualizer, PrintVisualizer

parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
parser.add_argument("--dev_sequences", default=1_000, type=int, help="Number of development sequences.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=0, type=int, help="Additional hidden layer after RNN.")
parser.add_argument("--logdir", default="logs", type=str, help="Directory for logs.")
parser.add_argument("--rnn", default="RNN", choices=["LSTM", "GRU", "RNN"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=8, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
parser.add_argument("--sequence_length", default=100, type=int, help="Sequence length.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_sequences", default=10_000, type=int, help="Number of training sequences.")
parser.add_argument("--device", default=None, type=str, help="Device override (cpu, cuda, mps).")
parser.add_argument("--log_file", default='logs.pkl', type=str, help="Log file")
# fmt: on


class ParitySequences:
    """Random integer sequences with prefix-sum parity labels."""

    def __init__(
        self, sequences_num: int, sequence_length: int, sequence_dim: int, seed: int
    ) -> None:
        sequences = torch.zeros(
            [sequences_num, sequence_length, sequence_dim], dtype=torch.int64
        )
        labels = torch.zeros([sequences_num, sequence_length, 1], dtype=torch.int64)
        generator = torch.Generator().manual_seed(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = torch.randint(
                0, max(2, sequence_dim), size=[sequence_length], generator=generator
            )
            labels[i, :, 0] = torch.bitwise_and(sequences[i, :, 0].cumsum(0), 1)
            if sequence_dim > 1:
                sequences[i] = torch.nn.functional.one_hot(
                    sequences[i, :, 0], sequence_dim
                )
        self._sequences = sequences.to(torch.float32)
        self._labels = labels.to(torch.float32)

    @property
    def sequences(self) -> torch.Tensor:
        return self._sequences

    @property
    def labels(self) -> torch.Tensor:
        return self._labels


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        rnn_kwargs = dict(
            input_size=args.sequence_dim,
            hidden_size=args.rnn_dim,
            batch_first=True,
        )
        match args.rnn:
            case "RNN":
                self.rnn = nn.RNN(**rnn_kwargs)
            case "LSTM":
                self.rnn = nn.LSTM(**rnn_kwargs)
            case "GRU":
                self.rnn = nn.GRU(**rnn_kwargs)

        self.hidden = nn.Identity()
        preoutput_dim = args.rnn_dim
        if args.hidden_layer > 0:
            preoutput_dim = args.hidden_layer
            self.hidden = nn.Sequential(
                nn.Linear(args.rnn_dim, preoutput_dim),
                nn.ReLU(),
            )

        self.output = nn.Sequential(
            nn.Linear(preoutput_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        rnn_outputs, _ = self.rnn(inputs)
        flat = rnn_outputs.reshape(batch_size * seq_len, -1)
        return self.output(self.hidden(flat)).reshape(batch_size, seq_len, -1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(override: str | None) -> torch.device:
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        total_loss += loss.item() * y.numel()
        total_correct += ((preds >= 0.5).float() == y).sum().item()
        total_count += y.numel()
    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_gradient: float | None,
) -> dict[str, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_correct += ((preds >= 0.5).float() == y).sum().item()
        total_count += y.numel()
    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


def main(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)

    train_data = ParitySequences(
        args.train_sequences, args.sequence_length, args.sequence_dim, seed=42
    )
    dev_data = ParitySequences(
        args.dev_sequences, args.sequence_length, args.sequence_dim, seed=43
    )

    train_loader = DataLoader(
        TensorDataset(train_data.sequences, train_data.labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        TensorDataset(dev_data.sequences, dev_data.labels),
        batch_size=args.batch_size,
    )

    model = Model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(
        args.logdir, f"parity-{args.rnn}-{args.rnn_dim}-{int(time.time())}.log"
    )

    final_logs: dict[str, float] = {}

    parameters = [name for name, _ in model.named_parameters()]
    inspector = PyTorchInspector(
        module=model,
        lenses=[
            lens.ParameterGradientGeometry(parameters=["weight_ih_l0", "weight_hh_l0"]),
        ],
        visualizer=RecorderVisualizer(f"logs/{args.log_file}"),
    )
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, args.clip_gradient
        )
        dev_metrics = evaluate(model, dev_loader, loss_fn, device)
        elapsed = time.time() - t0

        line = (
            f"Epoch {epoch:3d}/{args.epochs} "
            f"[{elapsed:5.1f}s]  "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f}  "
            f"dev loss={dev_metrics['loss']:.4f} acc={dev_metrics['accuracy']:.4f}"
        )
        print(line)

        inspector.tick()

        final_logs = {
            "train:loss": train_metrics["loss"],
            "train:accuracy": train_metrics["accuracy"],
            "dev:loss": dev_metrics["loss"],
            "dev:accuracy": dev_metrics["accuracy"],
        }

    return {k: v for k, v in final_logs.items() if k.startswith("dev:")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
    PlayerVisualizer("logs/logs.pkl", PrintVisualizer()).playback()
