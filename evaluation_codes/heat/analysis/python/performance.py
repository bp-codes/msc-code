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
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def plot_style(string, use_greyscale=False):
    string = string.lower()
    # --- Colour logic ---
    if use_greyscale:
        colour = "0.6"
    else:
        if "precise" in string:
            colour = "#f4a3a3"
        elif ("cuda" in string or "sycl" in string or "opencl" in string or "hip" in string):
            if "cpu" in string:
                colour = "#e6f3aa"
            else:
                colour = "#a9d6a5"
        elif ("parallel" in string or "openmp" in string):
            colour = "#a8c9f0"
        elif "serial" in string:
            colour = "#f6c28b"
        else:
            colour = "grey"

    # --- Hatch logic ---
    if "32" in string:
        hatch = "///"   # 'xx', '...', '\\\\'
    else:
        hatch = None

    return colour, hatch


def parse_float(value) -> float:
    return float(value)



def parse_float_list(values) -> list[float]:
    if not isinstance(values, list):
        raise TypeError("Expected a list of numeric values")

    return [parse_float(value) for value in values]


def load_results(
    results_dir: Path
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

        run_type = data.get("type")
        if run_type is None:
            continue

        input = data.get("input")
        if input is None:
            continue

        device = input.get("device")
        if device is not None:
            run_type = run_type + "_" + device

        for field in ["time_total", "max_rss_kb"]:
            value = data.get(field)

            if value is None:
                continue

            try:
                grouped[run_type][field].append(parse_float(value))
            except (TypeError, ValueError):
                pass

        """
        method = data.get("method")
        if method is None:
            continue

        method = str(method)

        device = data.get("device")
        method = method + " " + device

        if(selected_operation != ""):
            operation = data.get("operation")
            if(selected_operation != operation):
                continue

        for field in ["iterations", "max_rss_kb", "time_per_iteration"]:
            value = data.get(field)

            if value is None:
                continue

            try:
                grouped[method][field].append(parse_float(value))
                if(field == "time_per_iteration"):
                    grouped[method]["iterations_per_second"].append(1.0 / parse_float(value))
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
        """
    return grouped




def plot_performance_runtime(
    grouped: dict[str, dict[str, list[float]]],
    analysis_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    methods = []
    means = []
    mins = []

    for method, metrics in sorted(grouped.items()):
        time_total = metrics.get("time_total")

        if not time_total:
            continue

        methods.append(method)
        means.append(statistics.mean(time_total))
        mins.append(min(time_total))

    if not methods:
        print("No iteration data available for performance plot.")
        return

    plot_horizontal_bar(
        labels=methods,
        values=means,
        xlabel="Run time/s",
        title=f"Heat2D Runtime",
        output_dir="analysis",
        output_file = f"heat2d_mean_runtime.png".replace(" ", "_"),
        width=8,
        height=6,
        use_greyscale=False  # or False
    )
    write_columns_to_csv("heat2d_mean_runtime.csv", methods, means, ("Method", "Mean"))

    plot_horizontal_bar(
        labels=methods,
        values=mins,
        xlabel="Run time/s",
        title=f"Heat2D Runtime",
        output_dir="analysis",
        output_file = f"heat2d_min_runtime.png".replace(" ", "_"),
        width=8,
        height=6,
        use_greyscale=False  # or False
    )
    write_columns_to_csv("heat2d_mean_runtime.csv", methods, mins, ("Method", "Min"))




def plot_performance_memory(
    grouped: dict[str, dict[str, list[float]]],
    analysis_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    methods = []
    means = []
    maxs = []

    for method, metrics in sorted(grouped.items()):
        max_rss_kb = metrics.get("max_rss_kb")

        if not max_rss_kb:
            continue

        methods.append(method)
        means.append(statistics.mean(max_rss_kb))
        maxs.append(min(max_rss_kb))

    if not methods:
        print("No iteration data available for performance plot.")
        return

    plot_horizontal_bar(
        labels=methods,
        values=means,
        xlabel="Max Memory (kb)",
        title=f"Heat2D Max Memory (kb)",
        output_dir="analysis",
        output_file = f"heat2d_mean_memory.png".replace(" ", "_"),
        width=8,
        height=6,
        use_greyscale=False  # or False
    )

    write_columns_to_csv("heat2d_mean_memory.csv", methods, means, ("Method", "Mean"))

    plot_horizontal_bar(
        labels=methods,
        values=maxs,
        xlabel="Max Memory (kb)",
        title=f"Heat2D Max Memory (kb)",
        output_dir="analysis",
        output_file = f"heat2d_max_memory.png".replace(" ", "_"),
        width=8,
        height=6,
        use_greyscale=False  # or False
    )

    write_columns_to_csv("heat2d_max_memory.csv", methods, means, ("Method", "Max"))



def plot_horizontal_bar(labels,
                        values,
                        xlabel,
                        title,
                        output_dir,
                        output_file,
                        width=8,
                        height=6,
                        use_greyscale=False):

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, output_file)

    # Sort descending
    pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels_sorted, values_sorted = zip(*pairs)

    # --- Build colours + hatches ---
    colours = []
    hatches = []

    for label in labels_sorted:
        l = label.lower()

        colour, hatch = plot_style(l)

        colours.append(colour)
        hatches.append(hatch)

    # Create figure
    plt.figure(figsize=(width, height))

    bars = plt.barh(labels_sorted, values_sorted,
                    color=colours,
                    edgecolor="black")

    # Apply hatches individually
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    plt.xlabel(xlabel)
    plt.title(title)

    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot: {plot_path}")


def write_columns_to_csv(
    filename: str | Path,
    column1: Sequence[Any],
    column2: Sequence[Any],
    headers: tuple[str, str] | None = None,
) -> None:
    if len(column1) != len(column2):
        raise ValueError("The two lists must have the same length")

    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        if headers is not None:
            writer.writerow(headers)

        writer.writerows(zip(column1, column2))


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

    grouped = load_results(results_dir)
    plot_performance_runtime(grouped, analysis_dir)
    plot_performance_memory(grouped, analysis_dir)

    """
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

    plot_difference_histograms(trial, grouped, analysis_dir, selected_operation)
    plot_performance(trial, grouped, analysis_dir, selected_operation)
    """


if __name__ == "__main__":
    main()
