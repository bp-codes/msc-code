#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def exe_from_command(cmd: str) -> str:
    first = cmd.strip().split()[0]
    if first.startswith("./"):
        first = first[2:]
    return Path(first).name


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def max_per_exe(data: dict):
    buckets = defaultdict(lambda: {"iters": [], "rss": []})

    for cmd, vals in data.items():
        exe = exe_from_command(cmd)
        buckets[exe]["iters"].extend(vals.get("iterations", []))
        buckets[exe]["rss"].extend(vals.get("max_rss_kbytes", []))

    out = {}
    for exe, b in buckets.items():
        out[exe] = {
            "max_iters": max(b["iters"]) if b["iters"] else np.nan,
            "max_rss": max(b["rss"]) if b["rss"] else np.nan,
        }
    return out


def plot(json_paths, title=None, outfile=None):
    per_file = []
    labels = []
    all_exes = set()

    for p in json_paths:
        d = load_json(p)
        m = max_per_exe(d)
        per_file.append(m)
        labels.append(p.stem)
        all_exes.update(m.keys())

    exes = sorted(all_exes)
    n_exe = len(exes)
    n_files = len(json_paths)

    iters = np.full((n_files, n_exe), np.nan)
    rss = np.full((n_files, n_exe), np.nan)

    for fi, m in enumerate(per_file):
        for ei, exe in enumerate(exes):
            if exe in m:
                iters[fi, ei] = m[exe]["max_iters"]
                rss[fi, ei] = m[exe]["max_rss"]

    x = np.arange(n_exe)
    bar_width = 0.8 / max(n_files, 1)

    fig, ax1 = plt.subplots(figsize=(max(10, n_exe * 1.2), 6))
    ax2 = ax1.twinx()

    # ---- Iterations: BAR plot (ax1) ----
    for fi in range(n_files):
        ax1.bar(
            x + fi * bar_width,
            iters[fi],
            width=bar_width,
            color="tab:blue",
            alpha=0.5,          # <-- lighter
            label=f"{labels[fi]} iterations",
        )

    # ---- Memory: LINE plot (ax2, green) ----
    for fi in range(n_files):
        ax2.plot(
            x + bar_width * (n_files - 1) / 2,
            rss[fi],
            marker="o",
            linestyle="-",
            color="green",
            label=f"{labels[fi]} max RSS",
        )

    ax1.set_xticks(x + bar_width * (n_files - 1) / 2)
    ax1.set_xticklabels(exes, rotation=25, ha="right")

    ax1.set_ylabel("Max iterations")
    ax2.set_ylabel("Max RSS (kbytes)")

    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5)

    if title:
        fig.suptitle(title)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+")
    ap.add_argument("--title")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    plot([Path(p) for p in args.json_files], args.title, args.output)


if __name__ == "__main__":
    main()
