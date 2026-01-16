#!/usr/bin/env python3
import re
import json
import argparse
from collections import defaultdict

RE_COMMAND = re.compile(r'^\s*Command being timed:\s*"(.*?)"\s*$')
RE_CPU = re.compile(r'^\s*Percent of CPU this job got:\s*([0-9]+(?:\.[0-9]+)?)%\s*$')
RE_RSS = re.compile(r'^\s*Maximum resident set size \(kbytes\):\s*([0-9]+)\s*$')

def split_csv_line(line: str):
    # Very simple CSV splitting (your lines don’t contain quoted commas)
    return [p.strip() for p in line.strip().split(",")]

def parse_text(text: str):
    # group key: full command string, e.g. ./serial.x 5.0 1000000 add
    data = defaultdict(lambda: {
        "percent_cpu": [],
        "max_rss_kbytes": [],
        "iterations": []
    })

    current_cmd = None
    pending_iterations = None  # CSV 4th value observed before command line

    for raw in text.splitlines():
        line = raw.rstrip("\n")

        # CSV line: starts with something like "Serial," or "OpenMP,"
        # We only care about the 4th field (index 3) if it exists and is numeric.
        if "," in line and not line.lstrip().startswith(("Command being timed:", "User time", "System time",
                                                        "Percent of CPU", "Elapsed", "Average", "Maximum",
                                                        "Major", "Minor", "Voluntary", "Involuntary",
                                                        "Swaps", "File system", "Socket", "Signals",
                                                        "Page size", "Exit status", "Serial computed")):
            parts = split_csv_line(line)
            if len(parts) >= 4:
                field4 = parts[3]
                if re.fullmatch(r'-?\d+', field4):
                    pending_iterations = int(field4)

        m = RE_COMMAND.match(line)
        if m:
            current_cmd = m.group(1).strip()
            # attach any CSV 4th value we saw “nearby” (typically right before the command line)
            if pending_iterations is not None:
                data[current_cmd]["iterations"].append(pending_iterations)
                pending_iterations = None
            continue

        if current_cmd is None:
            continue

        m = RE_CPU.match(line)
        if m:
            data[current_cmd]["percent_cpu"].append(float(m.group(1)))
            continue

        m = RE_RSS.match(line)
        if m:
            data[current_cmd]["max_rss_kbytes"].append(int(m.group(1)))
            continue

        # end of a timing block: reset command when we see Exit status (optional but helps)
        if line.strip().startswith("Exit status:"):
            current_cmd = None

    return data

def main():
    ap = argparse.ArgumentParser(description="Parse benchmark output into JSON grouped by command.")
    ap.add_argument("input", help="Path to input log file (use - for stdin)")
    ap.add_argument("-o", "--output", default="-", help="Output JSON path (default: stdout)")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    args = ap.parse_args()

    if args.input == "-":
        import sys
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    parsed = parse_text(text)

    # Turn defaultdict into a normal dict for JSON
    out_obj = {cmd: vals for cmd, vals in parsed.items()}

    out_json = json.dumps(out_obj, indent=args.indent, sort_keys=True)
    if args.output == "-":
        print(out_json)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json + "\n")

if __name__ == "__main__":
    main()
 
