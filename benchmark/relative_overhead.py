import csv

stages = ["init", "forward", "backward", "optim", "tick"]

with open("experiments/benchmark_results.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["config"]]

baseline = next(r for r in rows if r["config"] == "baseline")

configs = [r for r in rows if r["config"] != "baseline"]

headers = ["config"] + [f"{s}_rel_%" for s in stages] + ["total_rel_%"]
col_w = max(len(r["config"]) for r in configs) + 2

print(f"{'config':<{col_w}}" + "".join(f"{h:>14}" for h in headers[1:]))
print("-" * (col_w + 14 * len(headers[1:])))

out_path = "experiments/relative_overhead.csv"
results = []

for row in configs:
    vals = [row["config"]]
    for stage in stages:
        base = float(baseline[f"{stage}_ms"])
        overhead = float(row[f"{stage}_overhead_ms"])
        vals.append(f"{overhead / base * 100:.2f}" if base > 0 else "N/A")
    base_total = float(baseline["total_time_s"])
    vals.append(f"{float(row['total_overhead_s']) / base_total * 100:.2f}")
    results.append(vals)
    print(f"{vals[0]:<{col_w}}" + "".join(f"{v:>14}" for v in vals[1:]))

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(results)

print(f"\nSaved to {out_path}")
