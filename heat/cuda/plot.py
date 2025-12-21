#!/usr/bin/env python3
"""
Plot and animate 2-D heat equation CSV snapshots produced by heat_solver.cpp.

Usage examples:
  # Preview a single frame
  python plot_heat.py out/heat_*.csv --frame 0

  # Animate all frames interactively
  python plot_heat.py out/heat_*.csv --animate

  # Save MP4 (needs ffmpeg installed)
  python plot_heat.py out/heat_*.csv --animate --save movie.mp4

  # Save GIF (no ffmpeg required if pillow installed)
  python plot_heat.py out/heat_*.csv --animate --save movie.gif

Optional:
  --vmin/--vmax to fix color limits, --cmap to change colormap,
  --skip to stride through frames (e.g., --skip 5 uses every 5th frame).
"""
import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HEADER_RE = re.compile(
    r"#\s*t\s*=\s*([0-9eE\.\+\-]+).*?nx\s*=\s*(\d+).*?ny\s*=\s*(\d+).*?Lx\s*=\s*([0-9eE\.\+\-]+).*?Ly\s*=\s*([0-9eE\.\+\-]+)"
)

def read_snapshot(path: str) -> Tuple[np.ndarray, Optional[float], Optional[Tuple[int,int,float,float]]]:
    """Read one CSV snapshot. Returns (array ny×nx, time, (nx,ny,Lx,Ly)).

    The first line may be metadata starting with '#'.
    """
    t = None
    meta = None
    with open(path, "r") as f:
        first = f.readline()
        m = HEADER_RE.search(first)
        if m:
            t = float(m.group(1))
            nx = int(m.group(2))
            ny = int(m.group(3))
            Lx = float(m.group(4))
            Ly = float(m.group(5))
            meta = (nx, ny, Lx, Ly)
            # load the rest
            arr = np.loadtxt(f, delimiter=",", dtype=float)
        else:
            # no header; include the first line when loading
            f.seek(0)
            arr = np.loadtxt(f, delimiter=",", dtype=float)

    # Ensure 2D
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr, t, meta

def load_sequence(pattern: str, skip: int = 1) -> Tuple[List[np.ndarray], List[Optional[float]], Optional[Tuple[int,int,float,float]]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    fields, times = [], []
    meta_all = None

    for k, p in enumerate(paths):
        if skip > 1 and (k % skip) != 0:
            continue
        arr, t, meta = read_snapshot(p)
        fields.append(arr)
        times.append(t)
        if meta is not None:
            meta_all = meta  # keep last seen metadata

    return fields, times, meta_all

def main():
    ap = argparse.ArgumentParser(description="Plot / animate heat equation CSV snapshots")
    ap.add_argument("pattern", help="Glob pattern for CSV files, e.g. out/heat_*.csv")
    ap.add_argument("--frame", type=int, default=None, help="Plot only this frame index (after skip).")
    ap.add_argument("--animate", action="store_true", help="Animate the sequence.")
    ap.add_argument("--interval", type=int, default=40, help="Animation interval (ms) between frames.")
    ap.add_argument("--repeat", action="store_true", help="Loop animation.")
    ap.add_argument("--save", type=str, default=None, help="Filename to save animation (mp4 or gif).")
    ap.add_argument("--dpi", type=int, default=120, help="DPI for saved animation.")
    ap.add_argument("--vmin", type=float, default=None, help="Fixed color lower bound.")
    ap.add_argument("--vmax", type=float, default=None, help="Fixed color upper bound.")
    ap.add_argument("--cmap", type=str, default="inferno", help="Matplotlib colormap.")
    ap.add_argument("--skip", type=int, default=1, help="Use every Nth frame.")
    args = ap.parse_args()

    fields, times, meta = load_sequence(args.pattern, skip=args.skip)
    if not fields:
        raise RuntimeError("No frames loaded.")

    ny, nx = fields[0].shape
    Lx = Ly = None
    if meta is not None:
        nx_m, ny_m, Lx, Ly = meta
        # if grid sizes disagree, ignore extent
        if (nx_m, ny_m) != (nx, ny):
            Lx = Ly = None

    # color limits
    if args.vmin is None or args.vmax is None:
        data_concat = np.concatenate([f.ravel() for f in fields])
        auto_vmin = data_concat.min()
        auto_vmax = data_concat.max()
        vmin = args.vmin if args.vmin is not None else auto_vmin
        vmax = args.vmax if args.vmax is not None else auto_vmax
    else:
        vmin, vmax = args.vmin, args.vmax

    extent = None
    if Lx is not None and Ly is not None:
        extent = [0.0, Lx, 0.0, Ly]

    # Single frame plot
    if args.frame is not None and not args.animate:
        idx = max(0, min(args.frame, len(fields)-1))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(fields[idx], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=args.cmap, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Temperature")
        if times[idx] is not None:
            ax.set_title(f"Heat field (t = {times[idx]:.6g})")
        else:
            ax.set_title(f"Heat field (frame {idx})")
        ax.set_xlabel("x" if extent else "i")
        ax.set_ylabel("y" if extent else "j")
        ax.set_aspect("equal" if extent else "auto")
        fig.tight_layout()
        plt.show()
        return

    # Animation
    if args.animate:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(fields[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=args.cmap, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Temperature")
        ttl = ax.set_title("Heat field")
        ax.set_xlabel("x" if extent else "i")
        ax.set_ylabel("y" if extent else "j")
        if extent:
            ax.set_aspect("equal")
        fig.tight_layout()

        def update(i):
            im.set_data(fields[i])
            if times[i] is not None:
                ttl.set_text(f"Heat field (t = {times[i]:.6g})")
            else:
                ttl.set_text(f"Heat field (frame {i})")
            return (im, ttl)

        anim = FuncAnimation(fig, update, frames=len(fields), interval=args.interval, blit=False, repeat=args.repeat)

        if args.save:
            fname = args.save
            root, ext = os.path.splitext(fname)
            if ext.lower() == ".mp4":
                try:
                    anim.save(fname, fps=max(1, int(1000/args.interval)), dpi=args.dpi, writer="ffmpeg")
                except Exception as e:
                    print(f"[!] MP4 save failed ({e}). Do you have ffmpeg installed? Try saving a GIF instead.")
                    raise
            elif ext.lower() == ".gif":
                anim.save(fname, fps=max(1, int(1000/args.interval)), dpi=args.dpi, writer=PillowWriter())
            else:
                raise ValueError("Unknown extension for --save (use .mp4 or .gif)")
            print(f"Saved animation to: {fname}")
        else:
            plt.show()
        return

    # Default: if neither --frame nor --animate provided, show first frame
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(fields[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=args.cmap, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Temperature")
    if times[0] is not None:
        ax.set_title(f"Heat field (t = {times[0]:.6g})")
    else:
        ax.set_title("Heat field (first frame)")
    ax.set_xlabel("x" if extent else "i")
    ax.set_ylabel("y" if extent else "j")
    if extent:
        ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
 
