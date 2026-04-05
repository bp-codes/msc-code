#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict



def parse_float(value) -> float:
    return float(value)



def parse_float_list(values) -> list[float]:
    if not isinstance(values, list):
        raise TypeError("Expected a list of numeric values")

    return [parse_float(value) for value in values]



def load_precise_data(results_dir: Path) -> tuple[float, list[float]]:
    precise_file = results_dir / "precise.json"

    if not precise_file.exists():
        raise FileNotFoundError(f"Precise reference file not found: {precise_file}")

    with open(precise_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON root in {precise_file}")

    if "calculated_value" not in data:
        raise KeyError(f"Missing 'calculated_value' in {precise_file}")

    if "values" not in data:
        raise KeyError(f"Missing 'values' in {precise_file}")

    precise_value = parse_float(data["calculated_value"])
    precise_values = parse_float_list(data["values"])

    return precise_value, precise_values



def load_results(
    results_dir: Path,
    precise_value: float,
    precise_values: list[float]
) -> dict[str, dict[str, list[float]]]:

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "precise.json":
            continue

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

        values = data.get("values")
        if values is not None:
            try:
                method_values = parse_float_list(values)

                if len(method_values) != len(precise_values):
                    print(
                        f"Warning: length mismatch in {json_file}: "
                        f"{len(method_values)} vs {len(precise_values)}"
                    )
                else:
                    array_differences = [
                        method_value - precise_value_element
                        for method_value, precise_value_element in zip(
                            method_values,
                            precise_values
                        )
                    ]

                    grouped[method]["values_difference_vs_precise"].extend(
                        array_differences
                    )
                    grouped[method]["abs_values_difference_vs_precise"].extend(
                        abs(value) for value in array_differences
                    )

            except (TypeError, ValueError) as e:
                print(f"Warning: failed to parse 'values' in {json_file}: {e}")

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



def plot_difference_histograms(
    grouped: dict[str, dict[str, list[float]]],
    analysis_dir: Path,
    bins: int = 50
) -> None:
    import matplotlib.pyplot as plt

    analysis_dir.mkdir(parents=True, exist_ok=True)

    for method, metrics in sorted(grouped.items()):
        values = metrics.get("values_difference_vs_precise")

        if not values:
            continue

        safe_method = method.replace("/", "_").replace(" ", "_")

        plt.figure()
        plt.hist(values, bins=bins)

        plt.title(f"Values Difference vs Precise ({method})")
        plt.xlabel("Difference (value - precise_value)")
        plt.ylabel("Frequency")

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(analysis_dir / f"hist_{safe_method}.png")
        plt.close()



def plot_performance(
    grouped: dict[str, dict[str, list[float]]],
    analysis_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    methods = []
    means = []

    for method, metrics in sorted(grouped.items()):
        iterations = metrics.get("iterations")

        if not iterations:
            continue

        methods.append(method)
        means.append(statistics.mean(iterations))

    if not methods:
        print("No iteration data available for performance plot.")
        return

    plt.figure()

    plt.bar(methods, means)

    plt.title("Mean Iterations by Method")
    plt.xlabel("Method")
    plt.ylabel("Mean Iterations")

    plt.xticks(rotation=45, ha="right")

    plt.grid(True)
    plt.tight_layout()

    output_file = analysis_dir / "performance_iterations.png"
    plt.savefig(output_file)
    plt.close()

    print(f"Saved performance plot to {output_file}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse benchmark results")

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Directory containing result JSON files"
    )

    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("analysis"),
        help="Directory to store analysis outputs"
    )

    args = parser.parse_args()

    results_dir = args.results
    analysis_dir = args.analysis

    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    analysis_dir.mkdir(parents=True, exist_ok=True)

    precise_value, precise_values = load_precise_data(results_dir)
    grouped = load_results(results_dir, precise_value, precise_values)

    if not grouped:
        print("No valid results found.")
        return

    print(f"Precise reference value: {precise_value:.17g}")

    iterations_summary = build_summary(grouped, "iterations")
    rss_summary = build_summary(grouped, "max_rss_kb")
    calculated_summary = build_summary(grouped, "calculated_value")
    difference_summary = build_summary(grouped, "difference_vs_precise")
    abs_difference_summary = build_summary(grouped, "abs_difference_vs_precise")
    values_difference_summary = build_summary(grouped, "values_difference_vs_precise")
    abs_values_difference_summary = build_summary(
        grouped,
        "abs_values_difference_vs_precise"
    )

    print_table("Iterations Statistics", iterations_summary)
    print_table("Max RSS (KB) Statistics", rss_summary)
    print_table("Calculated Value Statistics", calculated_summary)
    print_table("Difference vs Precise Statistics", difference_summary)
    print_table("Absolute Difference vs Precise Statistics", abs_difference_summary)
    print_table(
        "Values Difference vs Precise Statistics",
        values_difference_summary
    )
    print_table(
        "Absolute Values Difference vs Precise Statistics",
        abs_values_difference_summary
    )

    plot_difference_histograms(grouped, analysis_dir)
    plot_performance(grouped, analysis_dir)


if __name__ == "__main__":
    main()