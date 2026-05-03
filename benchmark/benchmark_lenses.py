#!/usr/bin/env python3
"""
benchmark_lenses.py
-------------------
Measures the per-phase wall-clock overhead that each monitorch lens adds to a
training loop by training a small ViT on synthetic TinyImageNet-sized data
(64×64 images, 200 classes, batch size 32).

Timing strategy
---------------
* **Forward / backward / optimizer step** — torch.utils.benchmark.Timer with
  blocked_autorange (auto-selects rep count, handles CUDA events internally).
* **Tick** — perf_counter + CUDA sync measured inline over 20 populate→tick
  cycles so the preprocessor is always in a realistic non-empty state, and
  correlation lenses are in steady state (two epochs seen).
* **Total wall-clock** — perf_counter around a fresh n_total_steps loop.
* **Initialization** — perf_counter around inspector construction + attach.

Configurations
--------------
  baseline                          – no inspector
  LossMetrics                       – inplace / inmemory
  OutputActivation                  – inplace / inmemory
  OutputGradientGeometry            – inplace × {corr, nocorr}
                                      inmemory × {corr, nocorr}
  OutputNorm                        – inplace / inmemory
  ParameterGradientActivation       – inplace / inmemory
  ParameterGradientGeometry         – inplace × {corr, nocorr}
                                      inmemory × {corr, nocorr}
  ParameterNorm                     – inplace / inmemory
  ParameterUpdateGeometry           – inplace × {corr, nocorr}
                                      inmemory × {corr, nocorr}

Output: benchmark_results.csv
"""

import argparse
import csv
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.utils.benchmark as tbenchmark

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

    def plot_numerical_values(self, epoch, main_tag, values_dict, ranges_dict=None) -> None:
        pass

    def plot_probabilities(self, epoch, main_tag, values_dict) -> None:
        pass

    def plot_relations(self, epoch, main_tag, values_dict) -> None:
        pass


# ---------------------------------------------------------------------------
# Small ViT — 64×64 input, 200 classes
#   patch 8×8 → 64 patches + cls token
#   embed_dim=256, depth=4, heads=4, mlp_dim=512  (~10M parameters)
#
# Layer types present (relevant for which lenses hook where):
#   nn.Conv2d    — PatchEmbed.proj
#   nn.Linear    — Attention.{qkv, proj}, MLP.{fc1, fc2}, head
#   nn.GELU      — MLP.act  (activation; picked up by OutputActivation / OutputNorm)
#   nn.LayerNorm — norm1, norm2, final norm  (has weight+bias; picked up by param lenses)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
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
    # Signature: (model, optimizer, criterion) -> list[AbstractLens]
    # The optimizer is passed so ParameterUpdateGeometry can register its hook.
    make_lenses: Callable


def build_configs() -> list[BenchConfig]:
    """
    Return one BenchConfig per (lens, inplace/inmemory[, corr/nocorr]) combination
    plus the no-lens baseline.
    """
    configs: list[BenchConfig] = []

    configs.append(BenchConfig('baseline', lambda m, opt, crit: []))

    for inplace in (True, False):
        tag = 'inplace' if inplace else 'inmemory'
        configs.append(
            BenchConfig(
                f'LossMetrics_{tag}',
                lambda m, opt, crit, ip=inplace: [LossMetrics(loss_fn=crit, loss_fn_inplace=ip)],
            )
        )

    for inplace in (True, False):
        tag = 'inplace' if inplace else 'inmemory'
        configs.append(
            BenchConfig(
                f'OutputActivation_{tag}',
                lambda m, opt, crit, ip=inplace: [OutputActivation(inplace=ip, channel_last=True)],
            )
        )

    for inplace in (True, False):
        for corr in (True, False):
            ip_tag = 'inplace' if inplace else 'inmemory'
            co_tag = 'corr' if corr else 'nocorr'
            configs.append(
                BenchConfig(
                    f'OutputGradientGeometry_{ip_tag}_{co_tag}',
                    lambda m, opt, crit, ip=inplace, c=corr: [OutputGradientGeometry(inplace=ip, compute_correlation=c)],
                )
            )

    for inplace in (True, False):
        tag = 'inplace' if inplace else 'inmemory'
        configs.append(
            BenchConfig(
                f'OutputNorm_{tag}',
                lambda m, opt, crit, ip=inplace: [OutputNorm(inplace=ip, channel_last=True)],
            )
        )

    for inplace in (True, False):
        tag = 'inplace' if inplace else 'inmemory'
        configs.append(
            BenchConfig(
                f'ParameterGradientActivation_{tag}',
                lambda m, opt, crit, ip=inplace: [ParameterGradientActivation(inplace=ip)],
            )
        )

    for inplace in (True, False):
        for corr in (True, False):
            ip_tag = 'inplace' if inplace else 'inmemory'
            co_tag = 'corr' if corr else 'nocorr'
            configs.append(
                BenchConfig(
                    f'ParameterGradientGeometry_{ip_tag}_{co_tag}',
                    lambda m, opt, crit, ip=inplace, c=corr: [ParameterGradientGeometry(inplace=ip, compute_correlation=c)],
                )
            )

    for inplace in (True, False):
        tag = 'inplace' if inplace else 'inmemory'
        configs.append(
            BenchConfig(
                f'ParameterNorm_{tag}',
                lambda m, opt, crit, ip=inplace: [ParameterNorm(inplace=ip)],
            )
        )

    for inplace in (True, False):
        for corr in (True, False):
            ip_tag = 'inplace' if inplace else 'inmemory'
            co_tag = 'corr' if corr else 'nocorr'
            configs.append(
                BenchConfig(
                    f'ParameterUpdateGeometry_{ip_tag}_{co_tag}',
                    lambda m, opt, crit, ip=inplace, c=corr: [ParameterUpdateGeometry(opt, inplace=ip, compute_correlation=c)],
                )
            )

    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _make_fresh(device: torch.device, seed: int, batch_size: int) -> tuple[SmallViT, torch.optim.Optimizer, nn.Module, torch.Tensor, torch.Tensor]:
    """Return a freshly initialised (model, optimizer, criterion, x, y)."""
    torch.manual_seed(seed)
    model = SmallViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    y = torch.randint(0, 200, (batch_size,), device=device)
    return model, optimizer, criterion, x, y


def _train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inspector: Optional[PyTorchInspector],
    x: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    tick_every: int,  # inspector.tick_epoch() every this many steps (0 = never)
    device: torch.device,
) -> float:
    """
    Run n_steps training steps, ticking the inspector every tick_every steps.
    Returns total wall-clock seconds.
    """
    _sync(device)
    t0 = time.perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if inspector is not None and tick_every > 0 and (step + 1) % tick_every == 0:
            inspector.tick_epoch()
    _sync(device)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def benchmark_config(
    cfg: BenchConfig,
    device: torch.device,
    batch_size: int,
    n_total_steps: int,
    n_warmup_steps: int,
    tick_every: int,
    seed: int,
) -> dict:
    """
    Run a single BenchConfig and return a flat dict of timing metrics (all in ms
    except total_time_s).

    Phase-timing detail
    -------------------
    forward
        Timer stmt: ``criterion(model(x), y)``
        Captures: FeedForwardGatherer hooks on model submodules (OutputActivation,
        OutputNorm) and on criterion (LossMetrics).

    backward
        Timer stmt: ``_loss.backward(retain_graph=True)``
        Pre-condition: a single loss tensor is computed once and its graph is kept
        alive with retain_graph.  Each Timer rep re-traverses the same graph.
        Captures: BackwardGatherer (register_full_backward_hook) and
        ParameterGradientGatherer (register_post_accumulate_grad_hook).

    optim_step
        Timer stmt: ``optimizer.step()``
        Pre-condition: gradients are pre-populated once via a real backward call
        and remain constant across Timer reps (zero_grad is intentionally NOT
        called between reps so the hook always fires with non-zero gradients).
        Captures: OptimizerStepParameterGatherer (register_step_post_hook).

    tick
        Measured inline: 20 cycles of (populate 100 steps → time one tick).
        A single discard tick is called first to prime correlation buffers.
        Captures: finalize_epoch + vizualize(NullVisualizer) + reset_epoch for
        all lenses, including EpochModuleGatherer calls in ParameterNorm.
    """
    model, optimizer, criterion, x, y = _make_fresh(device, seed, batch_size)

    _sync(device)
    t_init = time.perf_counter()
    lenses = cfg.make_lenses(model, optimizer, criterion)
    inspector: Optional[PyTorchInspector] = None
    if lenses:
        inspector = PyTorchInspector(
            lenses=lenses,
            module=model,
            visualizer=NullVisualizer(),
        )
    _sync(device)
    init_ms = (time.perf_counter() - t_init) * 1e3

    _train_steps(model, optimizer, criterion, inspector, x, y, n_warmup_steps, tick_every, device)

    t_fwd = tbenchmark.Timer(
        stmt='criterion(model(x), y)',
        globals={'model': model, 'criterion': criterion, 'x': x, 'y': y},
        num_threads=1,
        label=cfg.name,
        sub_label='forward',
    ).blocked_autorange(min_run_time=2.0)
    fwd_ms = t_fwd.median * 1e3

    optimizer.zero_grad()
    _retained_loss = criterion(model(x), y)
    t_bwd = tbenchmark.Timer(
        stmt='_loss.backward(retain_graph=True)',
        globals={'_loss': _retained_loss},
        num_threads=1,
        label=cfg.name,
        sub_label='backward',
    ).blocked_autorange(min_run_time=2.0)
    bwd_ms = t_bwd.median * 1e3
    del _retained_loss

    optimizer.zero_grad()
    criterion(model(x), y).backward()
    t_opt = tbenchmark.Timer(
        stmt='optimizer.step()',
        globals={'optimizer': optimizer},
        num_threads=1,
        label=cfg.name,
        sub_label='optim_step',
    ).blocked_autorange(min_run_time=2.0)
    opt_ms = t_opt.median * 1e3

    tick_samples: list[float] = []
    if inspector is not None:
        STEPS_PER_SAMPLE = 100  # steps to populate preprocessors between ticks
        N_SAMPLES = 20

        _train_steps(model, optimizer, criterion, inspector, x, y, STEPS_PER_SAMPLE, tick_every=0, device=device)
        inspector.tick_epoch()

        for _ in range(N_SAMPLES):
            _train_steps(model, optimizer, criterion, inspector, x, y, STEPS_PER_SAMPLE, tick_every=0, device=device)
            _sync(device)
            t0 = time.perf_counter()
            inspector.tick_epoch()
            _sync(device)
            tick_samples.append((time.perf_counter() - t0) * 1e3)

        tick_ms = statistics.median(tick_samples)
    else:
        tick_ms = 0.0

    # Re-create everything from the same seed so the starting state is identical
    # across all configs and the total time is directly comparable.
    model2, optimizer2, criterion2, x2, y2 = _make_fresh(device, seed, batch_size)
    lenses2 = cfg.make_lenses(model2, optimizer2, criterion2)
    inspector2: Optional[PyTorchInspector] = None
    if lenses2:
        inspector2 = PyTorchInspector(
            lenses=lenses2,
            module=model2,
            visualizer=NullVisualizer(),
        )

    total_time_s = _train_steps(
        model2,
        optimizer2,
        criterion2,
        inspector2,
        x2,
        y2,
        n_total_steps,
        tick_every,
        device,
    )

    del model, optimizer, inspector, model2, optimizer2, inspector2
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'config': cfg.name,
        'init_ms': round(init_ms, 4),
        'forward_ms': round(fwd_ms, 4),
        'backward_ms': round(bwd_ms, 4),
        'optim_ms': round(opt_ms, 4),
        'tick_ms': round(tick_ms, 4),
        'total_time_s': round(total_time_s, 4),
    }


# ---------------------------------------------------------------------------
# Post-processing: overhead columns relative to baseline
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    'config',
    'init_ms',
    'init_overhead_ms',
    'forward_ms',
    'forward_overhead_ms',
    'backward_ms',
    'backward_overhead_ms',
    'optim_ms',
    'optim_overhead_ms',
    'tick_ms',
    'tick_overhead_ms',
    'total_time_s',
    'total_overhead_s',
]


def add_overheads(rows: list[dict]) -> list[dict]:
    baseline = next(r for r in rows if r['config'] == 'baseline')
    out = []
    for row in rows:
        r = dict(row)
        r['init_overhead_ms'] = round(r['init_ms'] - baseline['init_ms'], 4)
        r['forward_overhead_ms'] = round(r['forward_ms'] - baseline['forward_ms'], 4)
        r['backward_overhead_ms'] = round(r['backward_ms'] - baseline['backward_ms'], 4)
        r['optim_overhead_ms'] = round(r['optim_ms'] - baseline['optim_ms'], 4)
        r['tick_overhead_ms'] = round(r['tick_ms'] - baseline['tick_ms'], 4)
        r['total_overhead_s'] = round(r['total_time_s'] - baseline['total_time_s'], 4)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Benchmark monitorch lens overhead on a small ViT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--steps', type=int, default=10_000, help='Training steps for the total wall-clock measurement.')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size (synthetic data; no actual dataset needed).')
    p.add_argument('--tick-every', type=int, default=500, help='Call inspector.tick_epoch() every N steps in the total loop.')
    p.add_argument('--warmup', type=int, default=200, help='Warmup steps before per-phase Timer measurements.')
    p.add_argument('--device', type=str, default='', help='PyTorch device string. Auto-detected if omitted.')
    p.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV path.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducible model init and data.')
    p.add_argument(
        '--include-inmemory',
        action='store_true',
        default=False,
        help=('Include inmemory configs even on CUDA. By default, inmemory benchmarks are skipped on CUDA because the deferred tensor aggregation in tick_epoch() forces a CPU-GPU sync, which inflates tick times and misrepresents realistic GPU training cost.'),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    configs = build_configs()

    # On CUDA, inmemory mode defers tensor aggregation to tick_epoch(), which
    # forces a CPU-GPU sync and badly inflates tick times.  Skip those configs
    # unless the user explicitly opts in with --include-inmemory.
    skip_inmemory = device.type == 'cuda' and not args.include_inmemory
    if skip_inmemory:
        configs = [c for c in configs if 'inmemory' not in c.name]

    print('=' * 60)
    print('monitorch lens benchmark')
    print('=' * 60)
    print(f'  device        : {device}')
    print(f'  steps         : {args.steps:,}')
    print(f'  batch size    : {args.batch_size}')
    print(f'  warmup        : {args.warmup}')
    print(f'  tick every    : {args.tick_every} steps')
    print(f'  configs       : {len(configs)}')
    if skip_inmemory:
        print(f'  inmemory      : skipped (CUDA — pass --include-inmemory to enable)')
    print(f'  output        : {args.output}')
    print('=' * 60)

    rows: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        print(f'\n[{i:2d}/{len(configs)}] {cfg.name}', flush=True)
        result = benchmark_config(
            cfg,
            device=device,
            batch_size=args.batch_size,
            n_total_steps=args.steps,
            n_warmup_steps=args.warmup,
            tick_every=args.tick_every,
            seed=args.seed,
        )
        rows.append(result)
        print(f'         total={result["total_time_s"]:.2f}s  fwd={result["forward_ms"]:.3f}ms  bwd={result["backward_ms"]:.3f}ms  opt={result["optim_ms"]:.3f}ms  tick={result["tick_ms"]:.3f}ms  init={result["init_ms"]:.2f}ms')

    rows = add_overheads(rows)

    out_path = Path(args.output)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nResults written to {out_path.resolve()}')

    # Pretty-print the overhead summary table
    print()
    header = f'{"config":<45} {"fwd_oh":>8} {"bwd_oh":>8} {"opt_oh":>8} {"tick_oh":>9} {"total_oh":>10}'
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["config"]:<45} {r["forward_overhead_ms"]:>+8.3f} {r["backward_overhead_ms"]:>+8.3f} {r["optim_overhead_ms"]:>+8.3f} {r["tick_overhead_ms"]:>+9.3f} {r["total_overhead_s"]:>+10.3f}')
    print()
    print('Units: overhead columns in ms (tick) / ms (fwd, bwd, opt) / s (total)')


if __name__ == '__main__':
    main()

