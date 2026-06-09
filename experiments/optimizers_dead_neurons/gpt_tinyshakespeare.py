import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
import urllib.request
import os
import tiktoken
from optimizer_utils import get_optimizer

from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import RecorderVisualizer
from monitorch import lens


# --- Data Preparation with Tiktoken ---
def get_batch_fn(block_size, batch_size, device):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("input.txt"):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, "input.txt")

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize the GPT-2 BPE Tokenizer
    print("Encoding dataset with tiktoken (GPT-2 BPE)...")
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    vocab_size = enc.n_vocab  # 50,257

    n = int(0.9 * len(data))
    train_data = data[:n]  # 90% train

    def get_batch():
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i : i + block_size] for i in ix])
        y = torch.stack([train_data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    return get_batch, vocab_size


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        # 'embed' and 'head' keywords route these to AdamW
        self.token_embed = nn.Embedding(vocab_size, config["n_embd"])
        self.position_embed = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.Sequential(
            *[
                Block(
                    config["n_embd"],
                    config["n_head"],
                    config["block_size"],
                    config["dropout"],
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos_emb = self.position_embed(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


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
    print(f"Running {config['experiment']} on {device}")

    get_batch, vocab_size = get_batch_fn(
        config["block_size"], config["batch_size"], device
    )
    print(f"Vocabulary Size: {vocab_size}")

    model = NanoGPT(vocab_size, config).to(device)
    optimizer = get_optimizer(
        model, config["optimizer"], config["lr"], config["weight_decay"]
    )

    log_file = args.config_file.split("/")[-1].split(".")[0] + f"_{config['seed']}.pkl"
    inspector = PyTorchInspector(
        lenses=[
            lens.LossMetrics(loss_fn=model),
            lens.OutputActivation(),
            lens.ParameterGradientActivation(),
        ],
        module=model,
        visualizer=RecorderVisualizer(log_file),
    )

    model.train()
    for iter_idx in range(config["epochs"]):
        xb, yb = get_batch()

        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        inspector.tick()
        if iter_idx % 100 == 0:
            print(f"Iteration {iter_idx} | CrossEntropy Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
