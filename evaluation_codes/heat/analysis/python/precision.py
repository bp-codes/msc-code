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
from matplotlib.colors import TwoSlopeNorm

FRAMEWORKS = ["cuda", "cuda_32", "openmp", "openmp_32", "opencl", "opencl_32", "serial", "serial_32", "sycl", "sycl_32"]


def plot_style(string, use_greyscale=False):
    # --- Colour logic ---
    if use_greyscale:
        colour = "0.6"
    else:
        if "precise" in string:
            colour = "#f4a3a3"
        elif ("cuda" in string or "sycl" in string or "opencl" in string):
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


def plot_difference_histogram(difference, output_dir, output_file, title, plot_normal=True):

    # Flatten to 1D
    values = difference.ravel()

    # Remove NaN/inf if needed
    values = values[np.isfinite(values)]

    count = values.size

    if count > 0:

        mean = np.mean(values)
        std = np.std(values)

        if std <= 0:
            std = 1e-12

        # Histogram
        histogram_counts, histogram_bins = np.histogram(
            values,
            bins=100
        )

        # Colours
        colour, hatch = plot_style(title)
        line_colour = "black"

        plt.figure(figsize=(10, 6))

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

        for bar in bars:
            bar.set_hatch(hatch)

        if(plot_normal):

            # Normal distribution fit
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

        plt.title(title)
        plt.xlabel(
            "Relative percentage difference to reference data"
        )
        plt.ylabel("Density")
        plt.yscale("log")
        plt.grid(True)
        if(plot_normal):
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_file))
        plt.close()



def plot_heatmap(
    output_dir, 
    output_file, 
    temperature,
    x=None,
    y=None,
    title="Temperature",
    xlabel="x",
    ylabel="y",
    label="Temperature",
    figsize=(8, 6),
    show=False,
    vmax=None,
    vmin=None,
    cmap="inferno",
    symmetric=None

):
    """
    Plot a 2D heatmap.
    Parameters
    ----------
    x : 1D array
        x coordinates

    y : 1D array
        y coordinates

    temperature : 2D array
        Temperature values with shape:
        (len(y), len(x))

    file_name : str | None
        Save figure if provided

    show : bool
        Display interactively
    """

    h, w = temperature.shape

    if x is None:
        x = np.linspace(
            -w / 2,
            w / 2,
            w
        )

    if y is None:
        y = np.linspace(
            -h / 2,
            h / 2,
            h
        )

    fig, ax = plt.subplots(figsize=figsize)

    if(symmetric is not None and symmetric == True):
        v = np.abs(temperature).max()
        if(v == 0):
            v = 1.0e-10
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        image = ax.imshow(
            temperature,
            extent=[
                np.min(x),
                np.max(x),
                np.min(y),
                np.max(y)
            ],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm
        )

    else:

        if(vmax is None):
            image = ax.imshow(
                temperature,
                extent=[
                    np.min(x),
                    np.max(x),
                    np.min(y),
                    np.max(y)
                ],
                origin="lower",
                aspect="auto",
                cmap=cmap
            )
        else:
            image = ax.imshow(
                temperature,
                extent=[
                    np.min(x),
                    np.max(x),
                    np.min(y),
                    np.max(y)
                ],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    file_name = os.path.join(output_dir, output_file)
    if file_name:
        plt.savefig(file_name, dpi=300)
    if show:
        plt.show()
    plt.close()



def load_set(path):
    results = []
    for csv_file in sorted(path.glob("*.csv")):
        arr, t, meta = read_snapshot(csv_file)
        print(t)
        results.append((arr, t, meta))
    return results



def main() -> None:

    parser = argparse.ArgumentParser(
        description="Analyse benchmark results"
    )

    parser.add_argument(
        "--framework",
        type=Path,
        default=Path(""),
        help="Framework"
    )

    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=Path(""),
        help="CSV directory in framework directory"
    )

    parser.add_argument(
        "--plot_name",
        type=Path,
        default=Path(""),
        help="Plot Name"
    )

    args = parser.parse_args()

    print(f"Precision for {args.framework}")

    output_dir = "analysis"
    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    precise = Path("../serial/output")
    precise_results = load_set(precise)

    openmp_32 = Path(f"../{args.framework}/{args.csv_dir}")
    openmp_32_results = load_set(openmp_32)

    last_step = min(len(precise_results), len(openmp_32_results)) - 1

    differences = [
        100 * np.divide(
            precise_results[i][0]
            - openmp_32_results[i][0],
            precise_results[i][0],
            out=np.zeros_like(precise_results[i][0]),
            where=precise_results[i][0] != 0
        )
        for i in range(last_step)
    ]

    print(differences)
    
    vmax_list = [
        np.maximum(np.max(precise_results[i][0]), np.max(openmp_32_results[i][0]))
        for i in range(last_step)
    ]
    print(vmax_list)
    vmax = np.max(vmax_list)
    print(vmax)

    vmax_differences_list = [
        np.max(differences[i])
        for i in range(last_step)
    ]
    print(vmax_differences_list)
    vmax_differences = np.max(vmax_differences_list)
    print(vmax_differences)

    vmin_differences_list = [
        np.min(differences[i])
        for i in range(last_step)
    ]
    print(vmin_differences_list)
    vmin_differences = np.min(vmin_differences_list)
    print(vmin_differences)

    times = [precise_results[i][1] for i in range(len(precise_results))]

    step = 0
    while(step < last_step):
        time = times[step]
        plot_difference_histogram(differences[step], output_dir, f"precision_{args.plot_name}_difference_{step}.png",
                                  f"{args.plot_name} Relative Difference {args.plot_name} t={time}s", False)
        
        # Heatmap for difference
        plot_heatmap(output_dir=output_dir,
                    output_file=f"heatmap_{args.plot_name}_difference_{step}.png",
                    temperature=differences[step],
                    title=f"Temperature difference {args.plot_name} t={time}s",
                    label="Percentage", 
                    cmap="RdBu_r",
                    symmetric=True)
        
        # Heatmap for reference
        plot_heatmap(output_dir=output_dir,
                    output_file=f"heatmap_reference_{step}.png",
                    temperature=precise_results[step][0],
                    title=f"Temperature reference t={time}s",
                    label="Temperature", 
                    vmax=vmax)
        
        # Heatmap for this
        plot_heatmap(output_dir=output_dir,
                    output_file=f"heatmap_{args.plot_name}_{last_step-1}.png",
                    temperature=openmp_32_results[step][0],
                    title=f"Temperature {args.plot_name} t={time}s",
                    label="Temperature", 
                    vmax=vmax)

        step = step + 10

    time = times[last_step-1]
    plot_difference_histogram(differences[last_step-1], output_dir, f"precision_{args.plot_name}_difference_{last_step-1}.png",
                                f"{args.plot_name} Relative Difference {args.plot_name} t={time}s", False)

    # Heatmap for difference
    plot_heatmap(output_dir=output_dir,
                output_file=f"heatmap_{args.plot_name}_difference_{last_step-1}.png",
                temperature=differences[last_step-1],
                title=f"Temperature difference {args.plot_name} t={time}s",
                label="Percentage", 
                cmap="RdBu_r",
                symmetric=True)
    
    # Heatmap for reference
    plot_heatmap(output_dir=output_dir,
                output_file=f"heatmap_reference_{last_step-1}.png",
                temperature=precise_results[last_step-1][0],
                title=f"Temperature reference t={time}s",
                label="Temperature", 
                vmax=vmax)
    
    # Heatmap for this
    plot_heatmap(output_dir=output_dir,
                output_file=f"heatmap_{args.plot_name}_{last_step-1}.png",
                temperature=openmp_32_results[last_step-1][0],
                title=f"Temperature {args.plot_name} t={time}s",
                label="Temperature", 
                vmax=vmax)



if __name__ == "__main__":
    main()
 
