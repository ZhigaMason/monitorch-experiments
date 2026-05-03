#!/usr/bin/env python
# coding: utf-8

# # CIFAR-10
#
# Simple convolutional network trained on CIFAR-10. Skeleton ready for `monitorch` instrumentation.

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

torch.set_num_threads(16)


# In[6]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

BATCH_SIZE = 64
EPOCHS = 50
WEIGHT_DECAY = 5e-2
HIDDEN_DIMS = [64, 64, 64]
DATA_ROOT = "./data"
SEED = 42

torch.manual_seed(SEED)


# In[7]:


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

print(f"train: {len(train_set)} | test: {len(test_set)}")


# In[22]:


from collections import OrderedDict


class CNN(nn.Module):
    def __init__(self, in_dim=3, hidden_dims=HIDDEN_DIMS, num_classes=10, dropout=0):
        super().__init__()
        layers = OrderedDict()
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            if i > 0:
                layers[f"dropout_{i}"] = nn.Dropout(dropout)
            layers[f"lin_{i}"] = nn.Conv2d(prev, h, kernel_size=3, stride=2, padding=1)
            layers[f"relu_{i}"] = nn.ReLU(inplace=True)
            prev = h
        layers["flatten"] = nn.Flatten()
        layers["output"] = nn.Linear(
            prev * 32 * 32 // (4 ** len(hidden_dims)), num_classes
        )
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x)


# In[23]:


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


# In[ ]:


from monitorch.inspector import PyTorchInspector
from monitorch import lens
from monitorch.visualizer import RecorderVisualizer

criterion = nn.CrossEntropyLoss()

list_of_lenses = [
    ([lens.LossMetrics(loss_fn=criterion)], "losses"),
]

model = CNN(hidden_dims=[64, 64, 64], dropout=0.5).to(DEVICE)
model

lr = 5e-3
momentum = 0.9

optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, betas=(momentum, 0.99), weight_decay=WEIGHT_DECAY
)

inspectors = []

for lenses, name in list_of_lenses:
    inspector = PyTorchInspector(
        module=model,
        lenses=lenses,
        visualizer=RecorderVisualizer(f"logs/{name}_{lr}_{momentum}_646464.pkl"),
    )
    inspectors.append(inspector)

for epoch in tqdm(range(1, EPOCHS + 1)):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    te_loss, te_acc = evaluate(model, test_loader, criterion, DEVICE)
    for inspector in inspectors:
        inspector.tick()
    print(
        f"epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}"
    )


# In[71]:


from monitorch.visualizer import PlayerVisualizer, MatplotlibVisualizer, PrintVisualizer

player = PlayerVisualizer(
    f"logs/{name}_{lr}_{momentum}_646464.pkl", PrintVisualizer()
).playback()

