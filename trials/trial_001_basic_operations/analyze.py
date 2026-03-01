#!/usr/bin/env python3
"""
python3 analyze.py --results "results/*.json" --targets target_values.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from typing import List, Dict, Any
from typing import Any, Dict, Iterable, List, Tuple
import matplotlib.pyplot as plt

operations_list = ["add", "divide", "exp", "log", "multiply", "power", "sqrt"]


def load_targets(directory):
    target_values_json = os.path.join(directory, "target_values.json")
    targets = {}

    with open(target_values_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    targets["checksum"] = data

    for op in operations_list:
        target_values_json = os.path.join(directory, "serial_stl_" + op + ".json")
        with open(target_values_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        targets[op] = data["values"]


    return targets



def load_results(directory):
    results = []
    pattern = os.path.join(directory, "*.json")
    for filepath in glob.glob(pattern):
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    return results



def process(
    results: List[Dict[str, Any]],
    target_values: Dict[str, float],
) -> Dict[str, Dict[str, Dict[str, float]]]:

    grouped_iters = defaultdict(lambda: defaultdict(list))
    grouped_pct_err = defaultdict(lambda: defaultdict(list))

    # --- group ---
    for r in results:
        operation = r.get("operation")
        method = r.get("method")
        iterations = r.get("iterations")
        calculated_value = r.get("calculated_value")

        if not isinstance(operation, str):
            continue
        if not isinstance(method, str):
            continue
        if not isinstance(iterations, int):
            continue
        if not isinstance(calculated_value, (int, float)):
            continue
        if operation not in target_values["checksum"]:
            continue

        target = target_values["checksum"][operation]

        # avoid divide-by-zero
        if target == 0:
            continue

        pct_error = ((calculated_value - target) / target) * 100.0

        grouped_iters[operation][method].append(iterations)
        grouped_pct_err[operation][method].append(pct_error)

    # --- compute stats ---
    processed = {}

    for operation, methods in grouped_iters.items():
        processed[operation] = {}

        for method, iters in methods.items():
            pct_errors = grouped_pct_err[operation][method]

            processed[operation][method] = {
                "min_iters": min(iters),
                "max_iters": max(iters),
                "mean_iters": statistics.fmean(iters),
                "median_iters": statistics.median(iters),
                "mean_percentage_error": statistics.fmean(pct_errors),
                "max_abs_percentage_error": max(abs(e) for e in pct_errors),
            }

    return processed



def save_json(processed, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)




def plot_iterations(
    processed: Dict[str, Dict[str, Dict[str, float]]],
    stat: str = "mean_iters",
    operations: Iterable[str] | None = None,
) -> None:
    """
    Plot iteration statistics per method, grouped by operation.

    Args:
        processed: output of process(), i.e. processed[operation][method][stat]
        stat: one of: "min_iters", "max_iters", "mean_iters", "median_iters"
        operations: optional list/iterable of operations to plot (default: all)
    """
    if operations is None:
        ops = sorted(processed.keys())
    else:
        ops = [op for op in operations if op in processed]

    for op in operations_list:
        methods = processed.get(op, {})
        if not methods:
            continue

        x_labels = sorted(methods.keys())
        y = [methods[m].get(stat) for m in x_labels]

        # drop missing
        pairs = [(m, v) for m, v in zip(x_labels, y) if isinstance(v, (int, float))]
        if not pairs:
            continue

        x_labels, y = zip(*pairs)

        plt.figure()
        plt.bar(range(len(x_labels)), y)
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
        plt.ylabel(stat)
        plt.title(f"{op} — {stat}")
        plt.tight_layout()
        plt.savefig(os.path.join("analysis", op + ".png"), dpi=200)
        plt.close()



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="glob for result JSON files")
    ap.add_argument("--targets", default="target_values", help="path to target_values")
    ap.add_argument("--outdir", default="analysis", help="output JSON summary file")
    ap.add_argument("--outjson", default="summary.json", help="output JSON summary file")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    target_values = load_targets(args.targets)
    results = load_results(args.results)
    processed = process(results, target_values)

    out_path = os.path.join(args.outdir, args.outjson)
    save_json(processed, out_path)

    print(processed)

    plot_iterations(processed, stat="median_iters", operations=["add"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 
