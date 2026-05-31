#!/usr/bin/env python3

from __future__ import annotations

import sys
import json
import statistics
from pathlib import Path


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


def load_functions(json_path: Path) -> list[dict[str, object]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {json_path}")

    return data


def summarise_project(json_path: Path) -> dict[str, object]:
    functions = load_functions(json_path)

    project_name = json_path.stem
    if project_name.startswith("complexity_"):
        project_name = project_name[len("complexity_"):]

    ccn_values = [int(item["ccn"]) for item in functions]
    nloc_values = [int(item["nloc"]) for item in functions]

    return {
        "project": project_name,
        "functions": len(functions),
        "total_ccn": sum(ccn_values),
        "mean_ccn": statistics.mean(ccn_values) if ccn_values else 0.0,
        "max_ccn": max(ccn_values) if ccn_values else 0,
        "total_nloc": sum(nloc_values),
        "mean_nloc": statistics.mean(nloc_values) if nloc_values else 0.0,
        "max_nloc": max(nloc_values) if nloc_values else 0,
    }


def summarise_libs(txt_path: Path) -> dict[str, int]:
    total = 0
    resolved = 0
    missing = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if ".so" not in line and "linux-vdso" not in line and "ld-linux" not in line:
                continue

            total += 1

            if "=> not found" in line:
                missing += 1
            else:
                resolved += 1

    return {
        "dep_total": total,
        "dep_resolved": resolved,
        "dep_missing": missing,
    }


def attach_lib_summary(
    row: dict[str, object],
    json_path: Path,
) -> dict[str, object]:
    project = str(row["project"])
    libs_path = json_path.with_name(f"libs_{project}.txt")

    if libs_path.exists():
        row.update(summarise_libs(libs_path))
    else:
        row.update(
            {
                "dep_total": 0,
                "dep_resolved": 0,
                "dep_missing": 0,
            }
        )

    return row


def collect_json_files(args: list[str]) -> list[Path]:
    paths: list[Path] = []

    for arg in args:
        p = Path(arg)

        if p.is_dir():
            paths.extend(sorted(p.glob("complexity_*.json")))
        elif p.is_file():
            if p.suffix.lower() == ".json":
                paths.append(p)
        else:
            raise FileNotFoundError(f"Path not found: {p}")

    # remove duplicates while preserving order
    unique_paths: list[Path] = []
    seen: set[Path] = set()

    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)

    return unique_paths


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "project",
        "functions",
        "total_ccn",
        "mean_ccn",
        "max_ccn",
        "total_nloc",
        "mean_nloc",
        "max_nloc",
        "dep_total"
    ]

    widths = {header: len(header) for header in headers}

    formatted_rows: list[dict[str, str]] = []

    for row in rows:
        formatted = {
            "project": str(row["project"]),
            "functions": str(row["functions"]),
            "total_ccn": str(row["total_ccn"]),
            "mean_ccn": f'{row["mean_ccn"]:.2f}',
            "max_ccn": str(row["max_ccn"]),
            "total_nloc": str(row["total_nloc"]),
            "mean_nloc": f'{row["mean_nloc"]:.2f}',
            "max_nloc": str(row["max_nloc"]),
            "dep_total": str(row["dep_total"])
        }
        formatted_rows.append(formatted)

        for header in headers:
            widths[header] = max(widths[header], len(formatted[header]))

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))

    for row in formatted_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python summarise_complexity.py <file_or_dir> [more paths...]")
        return 1

    json_files = collect_json_files(sys.argv[1:])

    if not json_files:
        print("No JSON files found.")
        return 1

    rows: list[dict[str, object]] = []

    for json_path in json_files:
        row = summarise_project(json_path)
        row = attach_lib_summary(row, json_path)
        rows.append(row)

    rows.sort(key=lambda row: str(row["project"]))
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())