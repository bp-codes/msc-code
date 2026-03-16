#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import defaultdict


def parse_float(value) -> float:
    return float(value)


def load_precise_value(results_dir: Path) -> float:
    precise_file = results_dir / "precise_sum.json"

    if not precise_file.exists():
        raise FileNotFoundError(f"Precise reference file not found: {precise_file}")

    with open(precise_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON root in {precise_file}")

    if "calculated_value" not in data:
        raise KeyError(f"Missing 'calculated_value' in {precise_file}")

    return parse_float(data["calculated_value"])


def load_results(
    results_dir: Path,
    precise_value: float
) -> dict[str, dict[str, list[float]]]:

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {json_file}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        method = data.get("method")
        if method is None:
            continue

        method = str(method)

        for field in ["iterations", "max_rss_kb"]:
            value = data.get(field)

            if value is None:
                continue

            try:
                grouped[method][field].append(parse_float(value))
            except (TypeError, ValueError):
                pass

        calculated_value = data.get("calculated_value")
        if calculated_value is not None:
            try:
                calculated = parse_float(calculated_value)
                difference = calculated - precise_value
                abs_difference = abs(difference)

                grouped[method]["calculated_value"].append(calculated)
                grouped[method]["difference_vs_precise"].append(difference)
                grouped[method]["abs_difference_vs_precise"].append(abs_difference)
            except (TypeError, ValueError):
                pass

    return grouped


def summarise(values: list[float]) -> dict[str, float | int | None]:

    values_sorted = sorted(values)
    n = len(values_sorted)

    result: dict[str, float | int | None] = {
        "count": n,
        "mean": statistics.mean(values_sorted),
        "median": statistics.median(values_sorted),
        "min": min(values_sorted),
        "max": max(values_sorted),
        "range": max(values_sorted) - min(values_sorted),
    }

    if n > 1:
        result["stdev"] = statistics.stdev(values_sorted)
    else:
        result["stdev"] = 0.0

    return result


def print_table(title: str, summary: dict[str, dict]) -> None:

    print(f"\n{title}")
    print("=" * len(title))

    header = (
        f"{'Method':<40}"
        f"{'Count':>8}"
        f"{'Mean':>15}"
        f"{'Median':>15}"
        f"{'Min':>15}"
        f"{'Max':>15}"
        f"{'StdDev':>15}"
    )

    print(header)
    print("-" * len(header))

    for method, stats in sorted(summary.items()):
        print(
            f"{method:<40}"
            f"{stats['count']:>8}"
            f"{stats['mean']:>15.6e}"
            f"{stats['median']:>15.6e}"
            f"{stats['min']:>15.6e}"
            f"{stats['max']:>15.6e}"
            f"{stats['stdev']:>15.6e}"
        )


def build_summary(grouped, metric):

    summary = {}

    for method, metrics in grouped.items():
        values = metrics.get(metric)
        if values:
            summary[method] = summarise(values)

    return summary


def main() -> None:

    results_dir = Path("results")

    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    precise_value = load_precise_value(results_dir)
    grouped = load_results(results_dir, precise_value)

    if not grouped:
        print("No valid results found.")
        return

    print(f"Precise reference value: {precise_value:.17g}")

    iterations_summary = build_summary(grouped, "iterations")
    rss_summary = build_summary(grouped, "max_rss_kb")
    calculated_summary = build_summary(grouped, "calculated_value")
    difference_summary = build_summary(grouped, "difference_vs_precise")
    abs_difference_summary = build_summary(grouped, "abs_difference_vs_precise")

    print_table("Iterations Statistics", iterations_summary)
    print_table("Max RSS (KB) Statistics", rss_summary)
    print_table("Calculated Value Statistics", calculated_summary)
    print_table("Difference vs Precise Statistics", difference_summary)
    print_table("Absolute Difference vs Precise Statistics", abs_difference_summary)


if __name__ == "__main__":
    main()