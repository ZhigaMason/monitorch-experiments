import torch
from torch.optim import Optimizer


class MultiOptimizer(Optimizer):
    """
    Wrapper to handle stepping multiple optimizers simultaneously,
    inheriting from Optimizer to support PyTorch step hooks.
    """

    def __init__(self, optimizers):
        self.optimizers = optimizers

        # Aggregate all parameter groups from the sub-optimizers
        param_groups = []
        for opt in optimizers:
            param_groups.extend(opt.param_groups)

        # Initialize base Optimizer with empty defaults (sub-optimizers handle their own)
        super().__init__(param_groups, defaults={})

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Fire sub-optimizer steps
        for opt in self.optimizers:
            opt.step()

        return loss


class MuonConvWrapper(Optimizer):
    """
    Bridge to use PyTorch's native Muon with >=3D parameters (4D Convs),
    inheriting from Optimizer to support PyTorch step hooks.
    """

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        # Create 2D proxy parameters dynamically from the initialized param_groups
        self.view_params = []
        for group in self.param_groups:
            for p in group["params"]:
                self.view_params.append(torch.nn.Parameter(p.data.view(p.size(0), -1)))

        # Initialize native Muon on the 2D proxies
        self.muon = torch.optim.Muon(self.view_params, lr=lr)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Map gradients from the original ND params to the 2D proxies
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                vp = self.view_params[idx]
                if p.grad is not None:
                    vp.grad = p.grad.view_as(vp)
                else:
                    vp.grad = None
                idx += 1

        # 2. Execute the native Muon Newton-Schulz step
        self.muon.step()

        # 3. Copy the updated 2D data back into the original ND parameter tensors
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                vp = self.view_params[idx]
                p.data.copy_(vp.data.view_as(p))
                idx += 1

        return loss


def get_optimizer(model, opt_name, lr, weight_decay=0.01):
    """Routes parameters to the correct optimizer based on YAML config."""
    if opt_name == "SGD":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    elif opt_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif opt_name == "Muon":
        muon_2d_params = []
        muon_nd_params = []
        adamw_params = []

        for name, p in model.named_parameters():
            # Exclude embeddings, classifier heads, and final layers from Muon
            if "embed" in name or "head" in name or "fc" in name:
                adamw_params.append(p)
            elif p.ndim == 2:
                # Standard hidden layers (Linear matrices, LSTM weights)
                muon_2d_params.append(p)
            elif p.ndim > 2:
                # Convolutional Kernels (4D)
                muon_nd_params.append(p)
            else:
                # 1D parameters (Biases, LayerNorms, BatchNorms)
                adamw_params.append(p)

        optimizers = []

        if muon_2d_params:
            optimizers.append(torch.optim.Muon(muon_2d_params, lr=lr))
        if muon_nd_params:
            optimizers.append(MuonConvWrapper(muon_nd_params, lr=lr))
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=weight_decay)
            )

        return MultiOptimizer(optimizers)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
