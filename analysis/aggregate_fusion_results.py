"""
aggregate_fusion_results.py — Scan results/Stage2/*/result.json and build a summary table.

Usage:
    python aggregate_fusion_results.py --results_dir results/Stage2 --out results/Stage2/summary.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.results_dir)
    rows = []
    for result_json in sorted(root.glob("*/result.json")):
        run_name = result_json.parent.name
        with open(result_json) as f:
            r = json.load(f)
        rows.append({
            "run": run_name,
            "views": "+".join(r["views"]),
            "method": r["method"],
            "best_epoch": r["best_epoch"],
            "val_acc": round(r["best_val_acc"], 4),
            "test_acc": round(r["test_acc"], 4),
            "out_dim": r["fusion_out_dim"],
            "params_M": round(r["num_params"] / 1e6, 2),
            "time_min": round(r["total_time_sec"] / 60, 1),
        })

    if not rows:
        print(f"No result.json found under {root}")
        return

    # Print as table
    cols = ["run", "views", "method", "best_epoch", "val_acc", "test_acc",
            "out_dim", "params_M", "time_min"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    # sort by test_acc desc
    for r in sorted(rows, key=lambda x: -x["test_acc"]):
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

    # Save CSV
    out_path = args.out or str(root / "summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved CSV -> {out_path}")

    # Per-method best
    print("\nBest run per method (by test_acc):")
    by_method = {}
    for r in rows:
        m = r["method"]
        if m not in by_method or r["test_acc"] > by_method[m]["test_acc"]:
            by_method[m] = r
    for m, r in by_method.items():
        print(f"  {m:<12} -> {r['run']:<20} test={r['test_acc']:.4f} val={r['val_acc']:.4f}")


if __name__ == "__main__":
    main()
