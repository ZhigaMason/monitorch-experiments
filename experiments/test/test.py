import os
import sys

from collections import OrderedDict

try:
    import monitorch
except ImportError:
    # Install your required packages
    status = os.system("pip install monitorch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from monitorch.inspector import PyTorchInspector  # noqa: E402
from monitorch.lens import (  # noqa: E402
    LossMetrics,
    ParameterGradientActivation,
    ParameterUpdateGeometry,
)
from monitorch.visualizer import (  # noqa: E402
    RecorderVisualizer,
    MatplotlibVisualizer,
    PlayerVisualizer,
)
from tqdm import trange  # noqa: E402

print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

classificator = nn.Sequential(
    OrderedDict(
        [
            ("lin1", nn.Linear(10, 10)),
            ("relu", nn.ReLU()),
            ("lin2", nn.Linear(10, 2)),
        ]
    )
)

if (
    current_device := torch.accelerator.current_accelerator(check_available=True)
) is not None:
    classificator.to(current_device)

optimizer = torch.optim.AdamW(classificator.parameters())
loss_fn = nn.CrossEntropyLoss()

inspector = PyTorchInspector(
    module=classificator,
    lenses=[
        LossMetrics(loss_fn=loss_fn),
        ParameterGradientActivation(),
        ParameterUpdateGeometry(optimizer),
    ],
    visualizer=RecorderVisualizer("log.pkl"),
)

for epoch in trange(20):
    for batch in range(1024):
        x = torch.normal(mean=0, std=1, size=(256, 10), dtype=torch.float32)
        y = ((x * x).sum(dim=1) < 1).to(torch.int64)

        pred = classificator(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    inspector.tick()

# effective way to close log file
del inspector

player = PlayerVisualizer("log.pkl", MatplotlibVisualizer()).playback()

fig = player.visualizer.show_fig()

fig.savefig("mpl_visualization.png")
