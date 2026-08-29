#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os


HISTOGRAM_BINS = np.linspace(-1e-3, 1e-3, 201)


def parse_float(value) -> float:
    return float(value)


def parse_float_list(values) -> np.ndarray:
    if not isinstance(values, list):
        raise TypeError("Expected a list of numeric values")

    return np.asarray(values, dtype=np.float64)


def update_running_stats(stats: dict, values: np.ndarray) -> None:

    count = len(values)

    if count == 0:
        return

    stats["count"] += count
    stats["sum"] += float(np.sum(values))
    stats["sum_sq"] += float(np.sum(values * values))
    stats["min"] = min(stats["min"], float(np.min(values)))
    stats["max"] = max(stats["max"], float(np.max(values)))


def build_difference_summary(grouped):

    summary = {}

    for method, metrics in grouped.items():

        stats = metrics.get("difference_stats")

        if stats is None:
            continue

        count = stats["count"]

        if count == 0:
            continue

        mean = stats["sum"] / count

        variance = (
            stats["sum_sq"] / count
            - mean * mean
        )

        variance = max(variance, 0.0)

        stdev = np.sqrt(variance)

        summary[method] = {
            "count": count,
            "mean": mean,
            "median": float("nan"),
            "min": stats["min"],
            "max": stats["max"],
            "stdev": stdev,
        }

    return summary


def load_precise_data(
    results_dir: Path,
    selected_operation: str
) -> tuple[float, np.ndarray]:

    if selected_operation == "":
        raise ValueError("selected_operation must not be empty")

    file_name = f"precise_{selected_operation}_gemm.json"

    precise_file = results_dir / file_name

    if not precise_file.exists():
        raise FileNotFoundError(
            f"Precise reference file not found: {precise_file}"
        )

    with open(precise_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON root in {precise_file}")

    if "expected_value" not in data:
        raise KeyError(f"Missing 'expected_value' in {precise_file}")

    if "values" not in data:
        raise KeyError(f"Missing 'values' in {precise_file}")

    precise_value = parse_float(data["expected_value"])

    precise_values = parse_float_list(data["values"])

    del data

    return precise_value, precise_values


def load_results(
    results_dir: Path,
    precise_value: float,
    precise_values: np.ndarray,
    selected_operation: str
):

    grouped = defaultdict(lambda: defaultdict(list))

    for json_file in sorted(results_dir.glob("*.json")):

        if json_file.name.startswith("precise_"):
            continue

        if selected_operation not in json_file.name:
            continue

        print("processing " + json_file.name)

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

        device = data.get("device")

        method = f"{method} {device}"

        if selected_operation != "":

            operation = data.get("operation")

            if selected_operation != operation:
                continue

        for field in [
            "iterations",
            "max_rss_kb",
            "time_per_iteration"
        ]:

            value = data.get(field)

            if value is None:
                continue

            try:
                parsed = parse_float(value)

                grouped[method][field].append(parsed)

                if field == "time_per_iteration":
                    grouped[method]["iterations_per_second"].append(
                        1.0 / parsed
                    )

            except (TypeError, ValueError):
                pass

        calculated_value = data.get("calculated_value")

        if calculated_value is not None:

            try:
                calculated = parse_float(calculated_value)

                difference = calculated - precise_value

                abs_difference = abs(difference)

                grouped[method]["calculated_value"].append(
                    calculated
                )

                grouped[method]["difference_vs_precise"].append(
                    difference
                )

                grouped[method]["abs_difference_vs_precise"].append(
                    abs_difference
                )

            except (TypeError, ValueError):
                pass

        values = data.get("values")

        del data

        if values is not None:

            try:
                method_values = parse_float_list(values)

                del values

                if len(method_values) != len(precise_values):

                    print(
                        f"Warning: length mismatch in {json_file}: "
                        f"{len(method_values)} vs "
                        f"{len(precise_values)}"
                    )

                    del method_values

                else:

                    array_differences = (
                        method_values - precise_values
                    )

                    del method_values

                    if "histogram_bins" not in grouped[method]:

                        grouped[method]["histogram_bins"] = (
                            HISTOGRAM_BINS
                        )

                        grouped[method]["histogram_counts"] = (
                            np.zeros(
                                len(HISTOGRAM_BINS) - 1,
                                dtype=np.int64
                            )
                        )

                        grouped[method]["difference_stats"] = {
                            "count": 0,
                            "sum": 0.0,
                            "sum_sq": 0.0,
                            "min": float("inf"),
                            "max": float("-inf"),
                        }

                    counts, _ = np.histogram(
                        array_differences,
                        bins=grouped[method]["histogram_bins"]
                    )

                    grouped[method]["histogram_counts"] += counts

                    update_running_stats(
                        grouped[method]["difference_stats"],
                        array_differences
                    )

                    del array_differences

            except (TypeError, ValueError) as e:
                print(
                    f"Warning: failed to parse "
                    f"'values' in {json_file}: {e}"
                )

    return grouped


def summarise(values: list[float]):

    values_sorted = sorted(values)

    n = len(values_sorted)

    result = {
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


def print_table(title: str, summary: dict) -> None:

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
    trial: str,
    grouped,
    analysis_dir: Path,
    selected_operation: str,
    use_greyscale: bool = False,
    width=8,
    height=6,
) -> None:

    analysis_dir.mkdir(parents=True, exist_ok=True)

    for method, metrics in sorted(grouped.items()):

        histogram_counts = metrics.get("histogram_counts")

        histogram_bins = metrics.get("histogram_bins")

        stats = metrics.get("difference_stats")

        if histogram_counts is None:
            continue

        count = stats["count"]

        if count == 0:
            continue

        mean = stats["sum"] / count

        variance = (
            stats["sum_sq"] / count
            - mean * mean
        )

        variance = max(variance, 0.0)

        std = np.sqrt(variance)

        if std <= 0:
            std = 1e-12

        safe_method = (
            method
            .replace("/", "_")
            .replace(" ", "_")
        )

        if use_greyscale:
            colour = "0.6"
            line_colour = "0.2"

        else:

            if "precise" in method.lower():
                colour = "#f4a3a3"

            elif any(
                k in method.lower()
                for k in ["cuda", "sycl", "opencl"]
            ):
                colour = "#a9d6a5"

            elif "parallel" in method.lower():
                colour = "#a8c9f0"

            elif "serial" in method.lower():
                colour = "#f6c28b"

            else:
                colour = "grey"

            line_colour = "black"

        hatch = "///" if "32" in method else None

        plt.figure(figsize=(width, height))

        widths = np.diff(histogram_bins)

        density = (
            histogram_counts
            / (np.sum(histogram_counts) * widths)
        )

        bars = plt.bar(
            histogram_bins[:-1],
            density,
            width=widths,
            align="edge",
            color=colour,
            alpha=0.6,
            edgecolor="black"
        )

        if hatch:
            for bar in bars:
                bar.set_hatch(hatch)

        x = np.linspace(
            histogram_bins[0],
            histogram_bins[-1],
            500
        )

        pdf = (
            (1 / (std * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
        )

        plt.plot(
            x,
            pdf,
            color=line_colour,
            linewidth=2,
            label="Normal fit"
        )

        plt.axvline(
            mean,
            linestyle="--",
            linewidth=2,
            color=line_colour,
            label=f"Mean = {mean:.4e}"
        )

        plt.axvline(
            mean + std,
            linestyle=":",
            linewidth=2,
            color=line_colour,
            label=f"+1σ = {mean + std:.4e}"
        )

        plt.axvline(
            mean - std,
            linestyle=":",
            linewidth=2,
            color=line_colour,
            label=f"-1σ = {mean - std:.4e}"
        )

        plt.title(
            f"{trial} ({selected_operation}): "
            f"Values Difference vs Precise ({method})"
        )

        plt.xlabel(
            "Difference (value - precise_value)"
        )

        plt.ylabel("Density")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        file_name = (
            f"{trial}_{selected_operation}_{safe_method}.png"
            .replace(" ", "_")
        )

        plt.savefig(analysis_dir / file_name)

        plt.close()


def plot_performance(
    trial: str,
    grouped,
    analysis_dir: Path,
    selected_operation: str
) -> None:

    methods = []

    means = []

    maxs = []

    for method, metrics in sorted(grouped.items()):

        iterations_per_second = metrics.get(
            "iterations_per_second"
        )

        if not iterations_per_second:
            continue

        methods.append(method)

        means.append(
            statistics.mean(iterations_per_second)
        )

        maxs.append(
            max(iterations_per_second)
        )

    if not methods:
        print(
            "No iteration data available "
            "for performance plot."
        )
        return

    plot_horizontal_bar(
        labels=methods,
        values=means,
        xlabel="Mean Iterations/s",
        title=(
            f"{trial} ({selected_operation}): "
            f"Mean Iterations/s by Method"
        ),
        output_dir="analysis",
        output_file=(
            f"{trial}_{selected_operation}_"
            f"performance_mean_iterations_per_second.png"
            .replace(" ", "_")
        ),
        width=8,
        height=6,
        use_greyscale=False
    )

    plot_horizontal_bar(
        labels=methods,
        values=maxs,
        xlabel="Max Iterations/s",
        title=(
            f"{trial} ({selected_operation}): "
            f"Max Iterations/s by Method"
        ),
        output_dir="analysis",
        output_file=(
            f"{trial}_{selected_operation}_"
            f"performance_max_iterations_per_second.png"
            .replace(" ", "_")
        ),
        width=8,
        height=6,
        use_greyscale=False
    )


def plot_horizontal_bar(
    labels,
    values,
    xlabel,
    title,
    output_dir,
    output_file,
    width=8,
    height=6,
    use_greyscale=False
):

    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, output_file)

    pairs = sorted(
        zip(labels, values),
        key=lambda x: x[1],
        reverse=True
    )

    labels_sorted, values_sorted = zip(*pairs)

    colours = []

    hatches = []

    for label in labels_sorted:

        l = label.lower()

        if use_greyscale:
            colour = "0.6"

        else:

            if "precise" in l:
                colour = "#f4a3a3"

            elif (
                "cuda" in l
                or "sycl" in l
                or "opencl" in l
            ):

                if "cpu" in l:
                    colour = "#e6f3aa"

                else:
                    colour = "#a9d6a5"

            elif "parallel" in l:
                colour = "#a8c9f0"

            elif "serial" in l:
                colour = "#f6c28b"

            else:
                colour = "grey"

        colours.append(colour)

        if "32" in l:
            hatch = "///"
        else:
            hatch = None

        hatches.append(hatch)

    plt.figure(figsize=(width, height))

    bars = plt.barh(
        labels_sorted,
        values_sorted,
        color=colours,
        edgecolor="black"
    )

    for bar, hatch in zip(bars, hatches):

        if hatch:
            bar.set_hatch(hatch)

    plt.xlabel(xlabel)

    plt.title(title)

    plt.gca().invert_yaxis()

    plt.grid(
        axis='x',
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(plot_path)

    plt.close()

    print(f"Saved plot: {plot_path}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Analyse benchmark results"
    )

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

    parser.add_argument(
        "--trial",
        type=str,
        default="Trial",
        help="Trial Name"
    )

    parser.add_argument(
        "--selected_operation",
        type=str,
        default="",
        help=""
    )

    args = parser.parse_args()

    results_dir = args.results

    analysis_dir = args.analysis

    trial = args.trial

    selected_operation = args.selected_operation

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {results_dir}"
        )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    precise_value, precise_values = load_precise_data(
        results_dir,
        selected_operation
    )

    grouped = load_results(
        results_dir,
        precise_value,
        precise_values,
        selected_operation
    )

    del precise_values

    if not grouped:
        print("No valid results found.")
        return

    print(
        f"Precise reference value: "
        f"{precise_value:.17g}"
    )

    iterations_summary = build_summary(
        grouped,
        "iterations"
    )

    rss_summary = build_summary(
        grouped,
        "max_rss_kb"
    )

    calculated_summary = build_summary(
        grouped,
        "calculated_value"
    )

    difference_summary = build_summary(
        grouped,
        "difference_vs_precise"
    )

    abs_difference_summary = build_summary(
        grouped,
        "abs_difference_vs_precise"
    )

    values_difference_summary = (
        build_difference_summary(grouped)
    )

    print_table(
        "Iterations Statistics",
        iterations_summary
    )

    print_table(
        "Max RSS (KB) Statistics",
        rss_summary
    )

    print_table(
        "Calculated Value Statistics",
        calculated_summary
    )

    print_table(
        "Difference vs Precise Statistics",
        difference_summary
    )

    print_table(
        "Absolute Difference vs Precise Statistics",
        abs_difference_summary
    )

    print_table(
        "Values Difference vs Precise Statistics",
        values_difference_summary
    )

    plot_difference_histograms(
        trial,
        grouped,
        analysis_dir,
        selected_operation
    )

    plot_performance(
        trial,
        grouped,
        analysis_dir,
        selected_operation
    )


if __name__ == "__main__":
    main()