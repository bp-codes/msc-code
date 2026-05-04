#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class Atom:
    element: str
    x: float
    y: float
    z: float


@dataclass
class Frame:
    atoms: list[Atom]
    comment: str


def read_xyz_frames(path: Path) -> list[Frame]:
    text = path.read_text(encoding="utf-8").splitlines()
    frames: list[Frame] = []

    i = 0
    n_lines = len(text)

    while i < n_lines:
        while i < n_lines and not text[i].strip():
            i += 1

        if i >= n_lines:
            break

        try:
            n_atoms = int(text[i].strip())
        except ValueError as exc:
            raise ValueError(
                f"{path}: expected atom count at line {i + 1}, got {text[i]!r}"
            ) from exc

        if i + 1 >= n_lines:
            raise ValueError(f"{path}: missing comment line after atom count at line {i + 1}")

        comment = text[i + 1]
        start = i + 2
        end = start + n_atoms

        if end > n_lines:
            raise ValueError(
                f"{path}: incomplete frame starting at line {i + 1}; expected {n_atoms} atoms"
            )

        atoms: list[Atom] = []
        for line_no, line in enumerate(text[start:end], start=start + 1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}: bad atom line at line {line_no}: {line!r}")

            element = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: invalid coordinates at line {line_no}: {line!r}"
                ) from exc

            atoms.append(Atom(element, x, y, z))

        frames.append(Frame(atoms=atoms, comment=comment))
        i = end

    if not frames:
        raise ValueError(f"{path}: no frames found")

    return frames


def compute_frame_diffs(
    ref: Frame,
    test: Frame,
) -> list[tuple[int, str, float, float, float, float]]:
    """
    Returns:
        [(atom_index_1_based, element, dx, dy, dz, dr), ...]
    where error = test - reference
    """
    if len(ref.atoms) != len(test.atoms):
        raise ValueError(
            f"Frame atom counts differ: {len(ref.atoms)} vs {len(test.atoms)}"
        )

    diffs: list[tuple[int, str, float, float, float, float]] = []

    for idx, (a, b) in enumerate(zip(ref.atoms, test.atoms), start=1):
        if a.element != b.element:
            raise ValueError(
                f"Atom mismatch at index {idx}: {a.element} vs {b.element}"
            )

        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        dr = math.sqrt(dx * dx + dy * dy + dz * dz)
        diffs.append((idx, a.element, dx, dy, dz, dr))

    return diffs


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) * (v - mu) for v in values) / len(values))


def plot_histogram(
    values: list[float],
    xlabel: str,
    title: str,
    bins: int,
    output_path: Path,
    as_frequency: bool,
) -> None:
    plt.figure(figsize=(8, 5))

    if not values:
        plt.xlabel(xlabel)
        plt.ylabel("Frequency" if as_frequency else "Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    mu = mean(values)
    sigma = stddev(values)

    counts, bin_edges, _ = plt.hist(
        values,
        bins=bins,
        weights=([1.0 / len(values)] * len(values) if as_frequency else None),
        alpha=0.7,
    )

    bin_width = bin_edges[1] - bin_edges[0]
    x_min = bin_edges[0]
    x_max = bin_edges[-1]

    # Build x values for the overlaid normal curve
    n_points = 400
    x_values = [
        x_min + (x_max - x_min) * i / (n_points - 1)
        for i in range(n_points)
    ]

    if sigma > 0.0:
        inv_sigma_sqrt_2pi = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        pdf_values = [
            inv_sigma_sqrt_2pi * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
            for x in x_values
        ]

        # Scale PDF to match histogram y-axis
        if as_frequency:
            # histogram sums to 1, so multiply by bin width
            y_values = [pdf * bin_width for pdf in pdf_values]
            plt.ylabel("Frequency")
        else:
            # histogram is in counts, so multiply by N * bin width
            y_values = [pdf * len(values) * bin_width for pdf in pdf_values]
            plt.ylabel("Count")

        plt.plot(
            x_values,
            y_values,
            label=f"Normal fit\nμ={mu:.3e}, σ={sigma:.3e}",
        )
        plt.legend()
    else:
        plt.ylabel("Frequency" if as_frequency else "Count")

    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_atoms_over_time(
    series_by_atom: dict[int, list[float]],
    elements_by_atom: dict[int, str],
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))

    for atom_index, series in series_by_atom.items():
        element = elements_by_atom[atom_index]
        plt.plot(series, label=f"{atom_index}:{element}")

    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_atoms_component_over_time(
    series_by_atom: dict[int, list[float]],
    elements_by_atom: dict[int, str],
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))

    for atom_index, series in series_by_atom.items():
        element = elements_by_atom[atom_index]
        plt.plot(series, label=f"{atom_index}:{element}")

    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze_trajectory(
    reference_path: Path,
    test_path: Path,
    top_n: int,
    bins: int,
    outdir: Path,
    histogram_mode: str,
) -> None:
    ref_frames = read_xyz_frames(reference_path)
    test_frames = read_xyz_frames(test_path)

    if len(ref_frames) != len(test_frames):
        raise ValueError(
            f"Number of frames differs: {reference_path} has {len(ref_frames)}, "
            f"{test_path} has {len(test_frames)}"
        )

    n_frames = len(ref_frames)
    if n_frames == 0:
        raise ValueError("No frames found")

    outdir.mkdir(parents=True, exist_ok=True)

    all_frame_diffs = [
        compute_frame_diffs(ref_frame, test_frame)
        for ref_frame, test_frame in zip(ref_frames, test_frames)
    ]

    last_diffs = all_frame_diffs[-1]
    n_atoms = len(last_diffs)

    dx_vals = [d[2] for d in last_diffs]
    dy_vals = [d[3] for d in last_diffs]
    dz_vals = [d[4] for d in last_diffs]
    dr_vals = [d[5] for d in last_diffs]

    dx_abs = [abs(v) for v in dx_vals]
    dy_abs = [abs(v) for v in dy_vals]
    dz_abs = [abs(v) for v in dz_vals]

    mean_displacement = mean(dr_vals)
    rms_displacement = rms(dr_vals)
    std_displacement = stddev(dr_vals)
    max_displacement = max(dr_vals)

    mean_dx = mean(dx_vals)
    mean_dy = mean(dy_vals)
    mean_dz = mean(dz_vals)

    std_dx = stddev(dx_vals)
    std_dy = stddev(dy_vals)
    std_dz = stddev(dz_vals)

    mean_abs_dx = mean(dx_abs)
    mean_abs_dy = mean(dy_abs)
    mean_abs_dz = mean(dz_abs)

    std_abs_dx = stddev(dx_abs)
    std_abs_dy = stddev(dy_abs)
    std_abs_dz = stddev(dz_abs)

    max_abs_dx = max(dx_abs)
    max_abs_dy = max(dy_abs)
    max_abs_dz = max(dz_abs)

    worst_last = sorted(last_diffs, key=lambda t: t[5], reverse=True)[:top_n]
    worst_indices = [item[0] for item in worst_last]
    elements_by_atom = {item[0]: item[1] for item in worst_last}

    print("=== XYZ trajectory comparison ===")
    print(f"Reference file          : {reference_path}")
    print(f"Test file               : {test_path}")
    print(f"Frames                  : {n_frames}")
    print(f"Atoms per frame         : {n_atoms}")
    print()
    print(f"Reference last comment  : {ref_frames[-1].comment}")
    print(f"Test last comment       : {test_frames[-1].comment}")
    print()

    print("=== Last-frame summary ===")
    print(f"Mean displacement / Å   : {mean_displacement:.12e}")
    print(f"RMS displacement / Å    : {rms_displacement:.12e}")
    print(f"Std displacement / Å    : {std_displacement:.12e}")
    print(f"Max displacement / Å    : {max_displacement:.12e}")
    print(f"Mean dx / Å             : {mean_dx:.12e}")
    print(f"Mean dy / Å             : {mean_dy:.12e}")
    print(f"Mean dz / Å             : {mean_dz:.12e}")
    print(f"Std dx / Å              : {std_dx:.12e}")
    print(f"Std dy / Å              : {std_dy:.12e}")
    print(f"Std dz / Å              : {std_dz:.12e}")
    print(f"Mean |dx| / Å           : {mean_abs_dx:.12e}")
    print(f"Mean |dy| / Å           : {mean_abs_dy:.12e}")
    print(f"Mean |dz| / Å           : {mean_abs_dz:.12e}")
    print(f"Std |dx| / Å            : {std_abs_dx:.12e}")
    print(f"Std |dy| / Å            : {std_abs_dy:.12e}")
    print(f"Std |dz| / Å            : {std_abs_dz:.12e}")
    print(f"Max |dx| / Å            : {max_abs_dx:.12e}")
    print(f"Max |dy| / Å            : {max_abs_dy:.12e}")
    print(f"Max |dz| / Å            : {max_abs_dz:.12e}")
    print()

    print(f"=== Top {len(worst_last)} worst atoms in final frame ===")
    print(f"{'Index':>8} {'El':>4} {'dx / Å':>18} {'dy / Å':>18} {'dz / Å':>18} {'dr / Å':>18}")
    for idx, element, dx, dy, dz, dr in worst_last:
        print(f"{idx:8d} {element:>4s} {dx:18.12e} {dy:18.12e} {dz:18.12e} {dr:18.12e}")

    as_frequency = histogram_mode == "frequency"

    hist_dx_path = outdir / "hist_dx_angstrom.png"
    hist_dy_path = outdir / "hist_dy_angstrom.png"
    hist_dz_path = outdir / "hist_dz_angstrom.png"
    hist_dr_path = outdir / "hist_dr_angstrom.png"

    plot_histogram(
        dx_vals,
        xlabel="Signed x error, dx = test_x - reference_x (Å)",
        title="Histogram of signed x error in final frame",
        bins=bins,
        output_path=hist_dx_path,
        as_frequency=as_frequency,
    )
    plot_histogram(
        dy_vals,
        xlabel="Signed y error, dy = test_y - reference_y (Å)",
        title="Histogram of signed y error in final frame",
        bins=bins,
        output_path=hist_dy_path,
        as_frequency=as_frequency,
    )
    plot_histogram(
        dz_vals,
        xlabel="Signed z error, dz = test_z - reference_z (Å)",
        title="Histogram of signed z error in final frame",
        bins=bins,
        output_path=hist_dz_path,
        as_frequency=as_frequency,
    )
    plot_histogram(
        dr_vals,
        xlabel="Displacement magnitude, dr = sqrt(dx² + dy² + dz²) (Å)",
        title="Histogram of displacement magnitude in final frame",
        bins=bins,
        output_path=hist_dr_path,
        as_frequency=as_frequency,
    )

    dr_series_by_atom: dict[int, list[float]] = {idx: [] for idx in worst_indices}
    dx_series_by_atom: dict[int, list[float]] = {idx: [] for idx in worst_indices}
    dy_series_by_atom: dict[int, list[float]] = {idx: [] for idx in worst_indices}
    dz_series_by_atom: dict[int, list[float]] = {idx: [] for idx in worst_indices}

    for frame_diffs in all_frame_diffs:
        by_index = {item[0]: item for item in frame_diffs}
        for atom_index in worst_indices:
            _, _, dx, dy, dz, dr = by_index[atom_index]
            dx_series_by_atom[atom_index].append(dx)
            dy_series_by_atom[atom_index].append(dy)
            dz_series_by_atom[atom_index].append(dz)
            dr_series_by_atom[atom_index].append(dr)

    top_dr_path = outdir / "top_atoms_dr_over_time.png"
    top_dx_path = outdir / "top_atoms_dx_over_time.png"
    top_dy_path = outdir / "top_atoms_dy_over_time.png"
    top_dz_path = outdir / "top_atoms_dz_over_time.png"

    plot_top_atoms_over_time(
        dr_series_by_atom,
        elements_by_atom,
        output_path=top_dr_path,
        ylabel="Displacement magnitude (Å)",
        title=f"Final-frame top {len(worst_indices)} worst atoms tracked over time",
    )

    plot_top_atoms_component_over_time(
        dx_series_by_atom,
        elements_by_atom,
        output_path=top_dx_path,
        ylabel="Signed x error (Å)",
        title=f"Signed dx over time for final-frame top {len(worst_indices)} worst atoms",
    )

    plot_top_atoms_component_over_time(
        dy_series_by_atom,
        elements_by_atom,
        output_path=top_dy_path,
        ylabel="Signed y error (Å)",
        title=f"Signed dy over time for final-frame top {len(worst_indices)} worst atoms",
    )

    plot_top_atoms_component_over_time(
        dz_series_by_atom,
        elements_by_atom,
        output_path=top_dz_path,
        ylabel="Signed z error (Å)",
        title=f"Signed dz over time for final-frame top {len(worst_indices)} worst atoms",
    )

    summary_json_path = outdir / "comparison_summary.json"

    summary_data = {
        "reference_xyz_path": str(reference_path),
        "test_xyz_path": str(test_path),
        "outdir": str(outdir),
        "histogram_mode": histogram_mode,
        "bins": bins,
        "top_n": top_n,
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "reference_last_comment": ref_frames[-1].comment,
        "test_last_comment": test_frames[-1].comment,
        "last_frame_summary_angstrom": {
            "mean_displacement": mean_displacement,
            "rms_displacement": rms_displacement,
            "std_displacement": std_displacement,
            "max_displacement": max_displacement,
            "mean_dx": mean_dx,
            "mean_dy": mean_dy,
            "mean_dz": mean_dz,
            "std_dx": std_dx,
            "std_dy": std_dy,
            "std_dz": std_dz,
            "mean_abs_dx": mean_abs_dx,
            "mean_abs_dy": mean_abs_dy,
            "mean_abs_dz": mean_abs_dz,
            "std_abs_dx": std_abs_dx,
            "std_abs_dy": std_abs_dy,
            "std_abs_dz": std_abs_dz,
            "max_abs_dx": max_abs_dx,
            "max_abs_dy": max_abs_dy,
            "max_abs_dz": max_abs_dz,
        },
        "top_worst_atoms_final_frame_angstrom": [
            {
                "index": idx,
                "element": element,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "dr": dr,
            }
            for idx, element, dx, dy, dz, dr in worst_last
        ],
        "output_files": {
            "hist_dx_angstrom_png": str(hist_dx_path),
            "hist_dy_angstrom_png": str(hist_dy_path),
            "hist_dz_angstrom_png": str(hist_dz_path),
            "hist_dr_angstrom_png": str(hist_dr_path),
            "top_atoms_dr_over_time_png": str(top_dr_path),
            "top_atoms_dx_over_time_png": str(top_dx_path),
            "top_atoms_dy_over_time_png": str(top_dy_path),
            "top_atoms_dz_over_time_png": str(top_dz_path),
            "comparison_summary_json": str(summary_json_path),
        },
    }

    summary_json_path.write_text(
        json.dumps(summary_data, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== Output files ===")
    print(hist_dx_path)
    print(hist_dy_path)
    print(hist_dz_path)
    print(hist_dr_path)
    print(top_dr_path)
    print(top_dx_path)
    print(top_dy_path)
    print(top_dz_path)
    print(summary_json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two XYZ trajectories, analyze final-frame atom position errors, "
            "plot histograms in Å, and track the final-frame top-N worst atoms over time."
        )
    )
    parser.add_argument("reference", type=Path, help="Reference XYZ file, e.g. 64-bit output")
    parser.add_argument("test", type=Path, help="Test XYZ file, e.g. 32-bit output")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of final-frame worst atoms to track over time (default: 10)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bin count (default: 50)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("xyz_analysis"),
        help="Output directory for plots (default: xyz_analysis)",
    )
    parser.add_argument(
        "--histogram-mode",
        choices=["count", "frequency"],
        default="count",
        help="Histogram y-axis mode (default: count)",
    )

    args = parser.parse_args()

    analyze_trajectory(
        reference_path=args.reference,
        test_path=args.test,
        top_n=args.top,
        bins=args.bins,
        outdir=args.outdir,
        histogram_mode=args.histogram_mode,
    )


if __name__ == "__main__":
    main()


"""
python3 analysis.py ../serial/results/out.xyz ../openmp/results/out.xyz --outdir=openmp
python3 analysis.py ../serial/results/out.xyz ../sycl/results/out.xyz --outdir=sycl
python3 analysis.py ../serial/results/out.xyz ../serial_32/results/out.xyz --outdir=serial_32
python3 analysis.py ../serial/results/out.xyz ../sycl_32/results/out.xyz --outdir=sycl_32
python3 analysis.py ../serial/results/out.xyz ../openmp_32/results/out.xyz --outdir=openmp_32
"""