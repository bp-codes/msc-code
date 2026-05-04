#!/usr/bin/env python3

import argparse
import glob
import os
import re
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def ensure_parent_dir(file_path: str | Path) -> None:
    p = Path(file_path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def read_snapshot(path: str) -> Tuple[np.ndarray, Optional[float], Optional[Tuple[int, int, float, float]]]:
    """Read one CSV snapshot. Returns (array ny×nx, time, (nx,ny,Lx,Ly)).

    The first line may be metadata starting with '#'.
    """
    t = None
    meta = None

    with open(path, "r") as f:
        first = f.readline()

        # Case 1: metadata header
        if first.startswith("#"):
            # Example expected format:
            # # t=0.1 nx=100 ny=100 Lx=1.0 Ly=1.0
            parts = first[1:].strip().replace(",", " ").split()
            values = dict(p.split("=") for p in parts)

            t = float(values.get("t")) if "t" in values else None
            nx = int(values.get("nx")) if "nx" in values else None
            ny = int(values.get("ny")) if "ny" in values else None
            Lx = float(values.get("Lx")) if "Lx" in values else None
            Ly = float(values.get("Ly")) if "Ly" in values else None

            if None not in (nx, ny, Lx, Ly):
                meta = (nx, ny, Lx, Ly)

            arr = np.loadtxt(f, delimiter=",", dtype=float)

        # Case 2: no header
        else:
            f.seek(0)
            arr = np.loadtxt(f, delimiter=",", dtype=float)

    # Ensure 2D
    if arr.ndim == 1:
        arr = arr[None, :]

    return arr, t, meta



def build_histogram_map(differences, bins=50, range=None, density=False):
    """
    Convert list of 2D arrays into a 2D histogram map.

    Returns:
        H  : (time, bins) array
        edges : bin edges
    """
    # Flatten all values once to get consistent binning
    all_values = np.concatenate([d.ravel() for d in differences])
    max_abs = np.max(np.abs(all_values))
    range = (-max_abs, max_abs)

    if range is None:
        vmin, vmax = -np.max(np.abs(all_values)), np.max(np.abs(all_values))
    else:
        vmin, vmax = range

    edges = np.linspace(vmin, vmax, bins + 1)

    H = []
    for d in differences:
        hist, _ = np.histogram(d.ravel(), bins=edges, density=density)
        H.append(hist)

    return np.array(H), edges


def plot_histogram_map(H, edges, times=None):
    plt.figure()

    extent = [edges[0], edges[-1], 0, H.shape[0]]

    plt.imshow(np.log1p(H), aspect='auto', origin='lower', extent=extent)
    plt.colorbar(label="Count")

    plt.xlabel("Difference value")
    plt.ylabel("Time step" if times is None else "Time")

    if times is not None:
        plt.yticks(
            np.linspace(0, len(times)-1, min(6, len(times))),
            [f"{t:.2f}" for t in np.linspace(times[0], times[-1], min(6, len(times)))]
        )

    plt.title("Distribution of differences over time")
    plt.tight_layout()
    plt.show()



def load_set(path):
    results = []
    for csv_file in sorted(path.glob("*.csv")):
        arr, t, meta = read_snapshot(csv_file)
        print(t)
        results.append((arr, t, meta))
    return results

def main():
    precise = Path("../serial/output")
    precise_results = load_set(precise)

    openmp_32 = Path("../sycl_32/output")
    openmp_32_results = load_set(openmp_32)

    differences = [
        precise_results[i][0] - openmp_32_results[i][0]
        for i in range(len(precise_results))
    ]

    times = [precise_results[i][1] for i in range(len(precise_results))]

    H, edges = build_histogram_map(differences, bins=60)
    plot_histogram_map(H, edges, times)

if __name__ == "__main__":
    main()
 
