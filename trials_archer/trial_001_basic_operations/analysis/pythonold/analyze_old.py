#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# Stats helpers (no SciPy)
# -----------------------------
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    31: 2.040, 32: 2.037, 33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028, 37: 2.026, 38: 2.024, 39: 2.023, 40: 2.021,
    50: 2.009, 60: 2.000, 70: 1.994, 80: 1.990, 90: 1.987, 100: 1.984, 120: 1.980
}

def tcrit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df < 120:
        keys = sorted(_T95.keys())
        lo = max(k for k in keys if k <= df)
        hi = min(k for k in keys if k >= df)
        if lo == hi:
            return _T95[lo]
        return _T95[lo] + (_T95[hi] - _T95[lo]) * (df - lo) / (hi - lo)
    return 1.96

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])

def stdev_sample(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

def ci95_mean(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    m = mean(xs)
    s = stdev_sample(xs)
    half = tcrit_95(n - 1) * s / math.sqrt(n)
    return (m - half, m + half)

def summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    lo, hi = ci95_mean(xs)
    return {
        "n": float(len(xs)),
        "mean": mean(xs),
        "median": median(xs),
        "lower": min(xs),
        "upper": max(xs),
        "stddev": stdev_sample(xs),
        "ci_lower": lo,
        "ci_upper": hi,
    }


# -----------------------------
# Data extraction / grouping
# -----------------------------
def safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def infer_operation(rec: Dict[str, Any]) -> str:
    op = safe_get(rec, ["parameters", "operation"])
    if isinstance(op, str) and op:
        return op
    comments = safe_get(rec, ["program_metrics", "comments"])
    if isinstance(comments, str) and "operation:" in comments:
        return comments.split("operation:", 1)[1].strip() or "unknown"
    return "unknown"

@dataclass(frozen=True)
class GroupKey:
    operation: str
    comments: str  # keep full comments so you can slice later if needed

@dataclass
class ByAppAgg:
    max_rss_kb: List[float]
    iterations: List[float]
    value_error: List[float]  # calculated - expected

    def __init__(self) -> None:
        self.max_rss_kb = []
        self.iterations = []
        self.value_error = []

def load_records(files: List[str]) -> List[Dict[str, Any]]:
    all_recs: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        recs = data.get("results", [])
        if not isinstance(recs, list):
            raise ValueError(f"{fp}: 'results' is not a list")
        all_recs.extend([r for r in recs if isinstance(r, dict)])
    return all_recs

def group_by_example(records: List[Dict[str, Any]]) -> Dict[GroupKey, Dict[str, ByAppAgg]]:
    """
    Returns:
      groups[(operation, comments)][app] -> series lists
    """
    groups: Dict[GroupKey, Dict[str, ByAppAgg]] = {}

    for r in records:
        app = str(r.get("app", "unknown"))
        op = infer_operation(r)
        comments = safe_get(r, ["program_metrics", "comments"])
        if not isinstance(comments, str) or not comments:
            comments = "unknown"

        gk = GroupKey(operation=op, comments=comments)
        by_app = groups.setdefault(gk, {})
        agg = by_app.setdefault(app, ByAppAgg())

        rss = safe_get(r, ["system", "max_rss_kb"])
        if isinstance(rss, (int, float)):
            agg.max_rss_kb.append(float(rss))

        it = safe_get(r, ["program_metrics", "iterations"])
        if isinstance(it, (int, float)):
            agg.iterations.append(float(it))

        ev = safe_get(r, ["program_metrics", "expected_value"])
        cv = safe_get(r, ["program_metrics", "calculated_value"])
        if isinstance(ev, (int, float)) and isinstance(cv, (int, float)):
            agg.value_error.append(float(cv) - float(ev))

    return groups


# -----------------------------
# Plotting
# -----------------------------
def plot_metric_by_app(
    groups: Dict[GroupKey, Dict[str, ByAppAgg]],
    metric: str,
    ylabel: str,
    out_png: str,
    show_points: bool = False,
) -> None:
    """
    For each example group (operation/comments), create one figure showing apps on x-axis and
    mean±95%CI for the selected metric.

    metric in {"max_rss_kb", "iterations", "value_error"}.
    """
    # Make one plot per example group (keeps comparisons clean)
    # If you prefer all operations on one plot, say so and I’ll collapse it.
    for gk, by_app in sorted(groups.items(), key=lambda x: (x[0].operation, x[0].comments)):
        apps = sorted(by_app.keys())

        labels: List[str] = []
        means: List[float] = []
        yerr_low: List[float] = []
        yerr_high: List[float] = []
        raw_series: List[List[float]] = []

        for app in apps:
            series = getattr(by_app[app], metric)
            if not series:
                # keep slot but mark NaN
                labels.append(app)
                means.append(float("nan"))
                yerr_low.append(0.0)
                yerr_high.append(0.0)
                raw_series.append([])
                continue

            s = summarize(series)
            labels.append(app)
            means.append(s["mean"])
            yerr_low.append(s["mean"] - s["ci_lower"])
            yerr_high.append(s["ci_upper"] - s["mean"])
            raw_series.append(series)

        # Skip empty plots
        if all((not rs) for rs in raw_series):
            continue

        plt.figure()
        x = list(range(len(labels)))

        plt.errorbar(
            x, means,
            yerr=[yerr_low, yerr_high],
            fmt="o", capsize=3
        )

        if show_points:
            # overlay individual points (small jitter)
            for i, series in enumerate(raw_series):
                for j, v in enumerate(series):
                    plt.plot(i + (j - (len(series)-1)/2)*0.01, v, marker=".", linestyle="")

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by application (mean ± 95% CI)\noperation={gk.operation}")
        plt.tight_layout()

        # Include group key in filename if multiple groups
        safe_tag = (gk.operation + "_" + gk.comments).replace("/", "_").replace(" ", "_").replace(":", "_")
        base, ext = os.path.splitext(out_png)
        fname = f"{base}__{safe_tag}{ext}"
        plt.savefig(fname, dpi=200)
        plt.close()

def write_summary_csv(groups: Dict[GroupKey, Dict[str, ByAppAgg]], out_csv: str) -> None:
    import csv

    fields = [
        "operation", "comments", "app",
        "metric", "n",
        "mean", "median",
        "lower", "upper",
        "stddev",
        "ci_lower", "ci_upper",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for gk, by_app in sorted(groups.items(), key=lambda x: (x[0].operation, x[0].comments)):
            for app, agg in sorted(by_app.items()):
                for metric_name, series in [
                    ("max_rss_kb", agg.max_rss_kb),
                    ("iterations", agg.iterations),
                    ("value_error", agg.value_error),
                ]:
                    if not series:
                        continue

                    s = summarize(series)
                    w.writerow({
                        "operation": gk.operation,
                        "comments": gk.comments,
                        "app": app,
                        "metric": metric_name,
                        "n": int(s["n"]),
                        "mean": s["mean"],
                        "median": s["median"],
                        "lower": s["lower"],
                        "upper": s["upper"],
                        "stddev": s["stddev"],
                        "ci_lower": s["ci_lower"],
                        "ci_upper": s["ci_upper"],
                    })



# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze benchmark JSON files and produce plots by app.")
    ap.add_argument("json_files", nargs="+", help="One or more benchmark_results.json files")
    ap.add_argument("--outdir", default="analysis", help="Output directory")
    ap.add_argument("--show-points", action="store_true", help="Overlay individual run points (can get busy)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    records = load_records(args.json_files)
    groups = group_by_example(records)


    # --- write summary ---
    summary_path = os.path.join(args.outdir, "summary.csv")
    write_summary_csv(groups, summary_path)
    print(f"Wrote {summary_path}")


    # Plot 1: memory vs app
    plot_metric_by_app(
        groups,
        metric="max_rss_kb",
        ylabel="Max RSS (KB)",
        out_png=os.path.join(args.outdir, "plot_memory_by_app.png"),
        show_points=args.show_points,
    )

    # Plot 2: iterations vs app
    plot_metric_by_app(
        groups,
        metric="iterations",
        ylabel="Iterations",
        out_png=os.path.join(args.outdir, "plot_iterations_by_app.png"),
        show_points=args.show_points,
    )

    # Plot 3: numerical error vs app
    plot_metric_by_app(
        groups,
        metric="value_error",
        ylabel="Calculated − Expected",
        out_png=os.path.join(args.outdir, "plot_error_by_app.png"),
        show_points=args.show_points,
    )

    print(f"Wrote plots into: {args.outdir}")
    print("Note: If you have multiple operations/comments, you will get one PNG per group (suffix in filename).")


if __name__ == "__main__":
    main()
