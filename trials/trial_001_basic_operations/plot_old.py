#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect(data: dict):
    commands = list(data.keys())

    iters = []
    rss = []
    for cmd in commands:
        it = data[cmd].get("iterations", [])
        mr = data[cmd].get("max_rss_kbytes", [])
        iters.append([float(x) for x in it])
        rss.append([float(x) for x in mr])

    return commands, iters, rss


def plot_subplots(commands, iters, rss, title=None, outfile=None):
    x = np.arange(len(commands))

    it_mean = np.array([np.mean(v) if len(v) else np.nan for v in iters])
    rss_mean = np.array([np.mean(v) if len(v) else np.nan for v in rss])

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(commands) * 1.2), 8), sharex=True)

    # Iterations subplot
    ax = axes[0]
    ax.plot(x, it_mean, marker="o", linestyle="-", label="mean iterations")
    for i, vals in enumerate(iters):
        if vals:
            ax.scatter([x[i]] * len(vals), vals, marker="x", label="runs" if i == 0 else None)
    ax.set_ylabel("Iterations")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    # RSS subplot
    ax2 = axes[1]
    ax2.plot(x, rss_mean, marker="o", linestyle="-", label="mean max RSS (kB)")
    for i, vals in enumerate(rss):
        if vals:
            ax2.scatter([x[i]] * len(vals), vals, marker="x", label="runs" if i == 0 else None)
    ax2.set_ylabel("Max RSS (kbytes)")
    ax2.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(commands, rotation=25, ha="right")

    if title:
        fig.suptitle(title)

    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_twin_axes(commands, iters, rss, title=None, outfile=None):
    x = np.arange(len(commands))

    it_mean = np.array([np.mean(v) if len(v) else np.nan for v in iters])
    rss_mean = np.array([np.mean(v) if len(v) else np.nan for v in rss])

    fig, ax1 = plt.subplots(figsize=(max(10, len(commands) * 1.2), 5))

    # Left axis: iterations
    ax1.plot(x, it_mean, marker="o", linestyle="-", label="mean iterations")
    for i, vals in enumerate(iters):
        if vals:
            ax1.scatter([x[i]] * len(vals), vals, marker="x", label="iter runs" if i == 0 else None)
    ax1.set_ylabel("Iterations")
    ax1.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    # Right axis: RSS
    ax2 = ax1.twinx()
    ax2.plot(x, rss_mean, marker="s", linestyle="-", label="mean max RSS (kB)")
    for i, vals in enumerate(rss):
        if vals:
            ax2.scatter([x[i]] * len(vals), vals, marker="+", label="RSS runs" if i == 0 else None)
    ax2.set_ylabel("Max RSS (kbytes)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(commands, rotation=25, ha="right")

    if title:
        fig.suptitle(title)

    # Combine legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot iterations and memory (RSS) from benchmark JSON.")
    ap.add_argument("json_path", help="Path to results.json")
    ap.add_argument("--mode", choices=["subplots", "twin"], default="subplots",
                    help="subplots = two plots; twin = one plot with two y-axes")
    ap.add_argument("--title", default=None, help="Optional plot title")
    ap.add_argument("-o", "--output", default=None, help="Save to file instead of showing (e.g. out.png)")
    args = ap.parse_args()

    data = load_results(Path(args.json_path))
    commands, iters, rss = collect(data)

    if args.mode == "subplots":
        plot_subplots(commands, iters, rss, title=args.title, outfile=args.output)
    else:
        plot_twin_axes(commands, iters, rss, title=args.title, outfile=args.output)


if __name__ == "__main__":
    main()
