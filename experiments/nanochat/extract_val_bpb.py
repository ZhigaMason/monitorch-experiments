#!/usr/bin/env python3
"""Extract final val_bpb from each run's stdout file into a CSV."""

import re
from pathlib import Path

import pandas as pd

RUNS_DIR = Path(__file__).parent / "repeated_runs"
OUT_CSV = Path(__file__).parent / "val_bpb.csv"

NAME_RE = re.compile(
    r"^nanochat-(?P<config>.+)-seed(?P<seed>\d+)\.(?P<stream>[oe])(?P<jobid>\d+)$"
)
BPB_RE = re.compile(r"Step\s+(\d+)\s*\|\s*Validation bpb:\s*([0-9.]+)")


def main() -> None:
    records = []
    for path in sorted(RUNS_DIR.iterdir()):
        m = NAME_RE.match(path.name)
        if not m or m.group("stream") != "o":
            continue
        matches = BPB_RE.findall(path.read_text(errors="replace"))
        final_step, final_bpb = matches[-1] if matches else (None, None)
        records.append(
            {
                "config": m.group("config"),
                "seed": int(m.group("seed")),
                "jobid": m.group("jobid"),
                "final_step": int(final_step) if final_step else None,
                "final_val_bpb": float(final_bpb) if final_bpb else None,
                "file": path.name,
            }
        )

    df = pd.DataFrame(records).sort_values(["config", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(
        f"Wrote {len(df)} rows to {OUT_CSV} ({df['final_val_bpb'].isna().sum()} with no val_bpb found)"
    )


if __name__ == "__main__":
    main()
