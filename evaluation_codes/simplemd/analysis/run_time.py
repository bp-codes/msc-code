#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return None


def plot_chart(run_types, time_keys, stats, title, ylabel, output_path, log_scale=False):
    x_positions = list(range(len(run_types)))
    n_metrics = len(time_keys)

    bar_width = 0.8 / n_metrics

    plt.figure(figsize=(12, 6))

    for i, time_key in enumerate(time_keys):
        offsets = [x + (i - (n_metrics - 1) / 2) * bar_width for x in x_positions]
        heights = [stats[run_type].get(time_key, 0.0) for run_type in run_types]
        plt.bar(offsets, heights, width=bar_width, label=time_key)

    plt.xticks(x_positions, run_types)
    plt.title(title)
    plt.xlabel("Type")
    plt.ylabel(ylabel)
    plt.legend(title="Time metric")

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot -> {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group JSON timing results and save bar charts (linear + log)."
    )
    parser.add_argument("path", type=Path, help="Directory containing JSON files")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    args = parser.parse_args()

    directory = args.path

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    files = sorted(directory.rglob("*.json") if args.recursive else directory.glob("*.json"))

    if not files:
        print(f"No JSON files found in {directory.resolve()}")
        return

    grouped_times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for path in files:
        data = load_json(path)
        if data is None:
            continue

        run_type = data.get("type")
        time_block = data.get("time")

        if not isinstance(run_type, str):
            print(f"Warning: {path} missing valid 'type'")
            continue

        if not isinstance(time_block, dict):
            print(f"Warning: {path} missing valid 'time'")
            continue

        for time_key, value in time_block.items():
            if isinstance(value, (int, float)):
                grouped_times[run_type][time_key].append(float(value))

    if not grouped_times:
        print("No valid timing data found.")
        return

    run_types = sorted(grouped_times.keys())

    time_keys_set: set[str] = set()
    for run_type in run_types:
        time_keys_set.update(grouped_times[run_type].keys())
    time_keys = sorted(time_keys_set)

    mean_stats: dict[str, dict[str, float]] = defaultdict(dict)
    max_stats: dict[str, dict[str, float]] = defaultdict(dict)

    print("\nSummary")
    print("=======")

    for run_type in run_types:
        print(f"\nType: {run_type}")
        print("-" * (6 + len(run_type)))

        for time_key in time_keys:
            values = grouped_times[run_type].get(time_key, [])
            if not values:
                continue

            min_v = min(values)
            mean_v = statistics.fmean(values)
            max_v = max(values)

            mean_stats[run_type][time_key] = mean_v
            max_stats[run_type][time_key] = max_v

            print(
                f"{time_key:25s} "
                f"min = {min_v:12.6f}  "
                f"mean = {mean_v:12.6f}  "
                f"max = {max_v:12.6f}  "
                f"n = {len(values)}"
            )

    output_dir = Path("run_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Linear plots
    plot_chart(
        run_types,
        time_keys,
        mean_stats,
        "Mean Times by Type",
        "Time",
        output_dir / "mean_times.png",
        log_scale=False,
    )

    plot_chart(
        run_types,
        time_keys,
        max_stats,
        "Max Times by Type",
        "Time",
        output_dir / "max_times.png",
        log_scale=False,
    )

    # Log plots
    plot_chart(
        run_types,
        time_keys,
        mean_stats,
        "Mean Times by Type (Log Scale)",
        "Time (log)",
        output_dir / "mean_times_log.png",
        log_scale=True,
    )

    plot_chart(
        run_types,
        time_keys,
        max_stats,
        "Max Times by Type (Log Scale)",
        "Time (log)",
        output_dir / "max_times_log.png",
        log_scale=True,
    )


if __name__ == "__main__":
    main()


"""
python3 run_time.py ../results/ --recursive
"""