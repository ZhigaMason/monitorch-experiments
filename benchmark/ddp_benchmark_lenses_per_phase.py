#!/usr/bin/env python3
"""
benchmark_lenses_async.py
-------------------------
Measures the real-world, fully pipelined throughput (Steps/Sec) impact of
monitorch lenses. This script completely removes internal synchronization
to allow PyTorch and NCCL to overlap communication and computation naturally.

Timing strategy:
----------------
* Warmup: Run N steps to initialize CUDA caching allocator and DDP buckets.
* Sync: A single global barrier and CUDA sync before the timer starts.
* Async Loop: Run N steps (with regular ticks) purely asynchronously.
* Sync: A final global barrier and CUDA sync to ensure all operations finished.
* Total Time: End - Start.

Usage:
  torchrun --nproc_per_node=NUM_GPUS benchmark_lenses_async.py [args]
"""

import argparse
import csv
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from monitorch.inspector import PyTorchInspector
from monitorch.lens import (
    LossMetrics,
    OutputActivation,
    OutputGradientGeometry,
    OutputNorm,
    ParameterGradientActivation,
    ParameterGradientGeometry,
    ParameterNorm,
    ParameterUpdateGeometry,
)
from monitorch.visualizer import AbstractVisualizer, TagAttributes


# ---------------------------------------------------------------------------
# No-op visualizer — eliminates all rendering I/O from measurements
# ---------------------------------------------------------------------------


class NullVisualizer(AbstractVisualizer):
    def register_tags(self, main_tag: str, tag_attr: TagAttributes) -> None:
        pass

    def plot_numerical_values(
        self, epoch, main_tag, values_dict, ranges_dict=None
    ) -> None:
        pass

    def plot_probabilities(self, epoch, main_tag, values_dict) -> None:
        pass

    def plot_relations(self, epoch, main_tag, values_dict) -> None:
        pass


# ---------------------------------------------------------------------------
# Small ViT (Unchanged)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        num_classes: int = 200,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    name: str
    make_lenses: Callable


def build_configs() -> list[BenchConfig]:
    configs: list[BenchConfig] = []
    configs.append(BenchConfig("baseline", lambda m, opt, crit: []))

    for inplace in (True, False):
        tag = "inplace" if inplace else "inmemory"
        configs.append(
            BenchConfig(
                f"LossMetrics_{tag}",
                lambda m, opt, crit, ip=inplace: [
                    LossMetrics(loss_fn=crit, loss_fn_inplace=ip)
                ],
            )
        )
        configs.append(
            BenchConfig(
                f"OutputActivation_{tag}",
                lambda m, opt, crit, ip=inplace: [
                    OutputActivation(inplace=ip, channel_last=True)
                ],
            )
        )
        configs.append(
            BenchConfig(
                f"OutputNorm_{tag}",
                lambda m, opt, crit, ip=inplace: [
                    OutputNorm(inplace=ip, channel_last=True)
                ],
            )
        )
        configs.append(
            BenchConfig(
                f"ParameterGradientActivation_{tag}",
                lambda m, opt, crit, ip=inplace: [
                    ParameterGradientActivation(inplace=ip)
                ],
            )
        )
        configs.append(
            BenchConfig(
                f"ParameterNorm_{tag}",
                lambda m, opt, crit, ip=inplace: [ParameterNorm(inplace=ip)],
            )
        )

    for inplace in (True, False):
        for corr in (True, False):
            ip_tag = "inplace" if inplace else "inmemory"
            co_tag = "corr" if corr else "nocorr"
            configs.append(
                BenchConfig(
                    f"OutputGradientGeometry_{ip_tag}_{co_tag}",
                    lambda m, opt, crit, ip=inplace, c=corr: [
                        OutputGradientGeometry(inplace=ip, compute_correlation=c)
                    ],
                )
            )
            configs.append(
                BenchConfig(
                    f"ParameterGradientGeometry_{ip_tag}_{co_tag}",
                    lambda m, opt, crit, ip=inplace, c=corr: [
                        ParameterGradientGeometry(inplace=ip, compute_correlation=c)
                    ],
                )
            )
            configs.append(
                BenchConfig(
                    f"ParameterUpdateGeometry_{ip_tag}_{co_tag}",
                    lambda m, opt, crit, ip=inplace, c=corr: [
                        ParameterUpdateGeometry(opt, inplace=ip, compute_correlation=c)
                    ],
                )
            )

    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_sync(device: torch.device):
    """Enforces absolute synchronization across GPUs and CPU."""
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_fresh(device: torch.device, seed: int, batch_size: int, local_rank: int):
    # Model seed identical across ranks
    torch.manual_seed(seed)
    model = SmallViT().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Data seed unique per rank
    torch.manual_seed(seed + local_rank)
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    y = torch.randint(0, 200, (batch_size,), device=device)

    return ddp_model, optimizer, criterion, x, y


# ---------------------------------------------------------------------------
# Core Async Benchmark Loop
# ---------------------------------------------------------------------------


def benchmark_config_async(
    cfg: BenchConfig,
    device: torch.device,
    local_rank: int,
    batch_size: int,
    n_total_steps: int,
    n_warmup_steps: int,
    tick_every: int,
    seed: int,
) -> dict:

    ddp_model, optimizer, criterion, x, y = _make_fresh(
        device, seed, batch_size, local_rank
    )

    unwrapped_model = ddp_model.module
    lenses = cfg.make_lenses(unwrapped_model, optimizer, criterion)

    inspector: Optional[PyTorchInspector] = None
    if lenses:
        inspector = PyTorchInspector(
            lenses=lenses,
            module=unwrapped_model,
            visualizer=NullVisualizer(),
        )

    # 1. WARMUP (Compile kernels, init DDP buckets, fill buffers)
    for step in range(n_warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(ddp_model(x), y)
        loss.backward()
        optimizer.step()
        if inspector and tick_every > 0 and (step + 1) % tick_every == 0:
            inspector.tick_epoch()

    # 2. START EXACT MEASUREMENT
    _full_sync(device)
    t0 = time.perf_counter()

    # 3. ASYNC LOOP (No internal syncs/barriers allowed)
    for step in range(n_total_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(ddp_model(x), y)
        loss.backward()
        optimizer.step()

        if inspector and tick_every > 0 and (step + 1) % tick_every == 0:
            inspector.tick_epoch()

    # 4. END EXACT MEASUREMENT
    _full_sync(device)
    total_time_s = time.perf_counter() - t0

    throughput = n_total_steps / total_time_s

    # Cleanup
    del ddp_model, optimizer, inspector, unwrapped_model, lenses, loss, x, y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "config": cfg.name,
        "total_time_s": round(total_time_s, 4),
        "throughput": round(throughput, 2),
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

CSV_FIELDS = ["config", "total_time_s", "throughput", "throughput_drop_pct"]


def add_overheads(rows: list[dict]) -> list[dict]:
    baseline = next(r for r in rows if r["config"] == "baseline")
    base_throughput = baseline["throughput"]

    out = []
    for row in rows:
        r = dict(row)
        drop_pct = ((base_throughput - r["throughput"]) / base_throughput) * 100
        r["throughput_drop_pct"] = round(drop_pct, 2)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fully async throughput benchmark for monitorch lenses in DDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--steps", type=int, default=5_000, help="Measured steps (post-warmup)."
    )
    p.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU.")
    p.add_argument("--tick-every", type=int, default=500, help="Tick frequency.")
    p.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup steps to bypass JIT/DDP init overhead.",
    )
    p.add_argument(
        "--output", type=str, default="benchmark_async_results.csv", help="Output CSV."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--include-inmemory",
        action="store_true",
        default=False,
        help="Include inmemory configs on CUDA.",
    )
    return p.parse_args()


def main() -> None:
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        print("WARNING: Not running under torchrun. DDP requires torchrun/submitit.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )
        local_rank = 0
        global_rank = 0
        world_size = 1

    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    configs = build_configs()
    skip_inmemory = device.type == "cuda" and not args.include_inmemory
    if skip_inmemory:
        configs = [c for c in configs if "inmemory" not in c.name]

    if global_rank == 0:
        print("=" * 60)
        print("monitorch async throughput benchmark")
        print("=" * 60)
        print(f"  world_size    : {world_size}")
        print(f"  device        : {device.type}")
        print(f"  steps         : {args.steps:,} (excluding warmup)")
        print(f"  batch size    : {args.batch_size} (per GPU)")
        print(f"  tick every    : {args.tick_every} steps")
        print("=" * 60)

    rows: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        if global_rank == 0:
            print(f"\n[{i:2d}/{len(configs)}] {cfg.name}", flush=True)

        result = benchmark_config_async(
            cfg,
            device=device,
            local_rank=local_rank,
            batch_size=args.batch_size,
            n_total_steps=args.steps,
            n_warmup_steps=args.warmup,
            tick_every=args.tick_every,
            seed=args.seed,
        )
        rows.append(result)

        if global_rank == 0:
            print(
                f"         total={result['total_time_s']:.2f}s  |  throughput={result['throughput']:.2f} steps/s"
            )

    if global_rank == 0:
        rows = add_overheads(rows)
        out_path = Path(args.output)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nResults written to {out_path.resolve()}")

        print()
        header = f"{'config':<45} {'total_time_s':>12} {'steps/sec':>12} {'drop_%':>10}"
        print(header)
        print("-" * len(header))
        for r in rows:
            drop_str = (
                f"{r['throughput_drop_pct']:+5.2f}%"
                if r["config"] != "baseline"
                else "---"
            )
            print(
                f"{r['config']:<45} {r['total_time_s']:>12.2f} {r['throughput']:>12.2f} {drop_str:>10}"
            )
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
