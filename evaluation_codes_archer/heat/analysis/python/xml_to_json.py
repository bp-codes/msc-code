#!/usr/bin/env python3

from __future__ import annotations

import sys
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_item_name(name: str) -> dict[str, object]:
    match = re.match(r"^(.*?) at (.*?):(\d+)$", name)
    if match is None:
        return {
            "name": name,
            "file": None,
            "line": None,
        }

    return {
        "name": match.group(1),
        "file": match.group(2),
        "line": int(match.group(3)),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python xml_to_json.py <complexity.xml>")
        return 1

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() != ".xml":
        raise ValueError(f"Expected .xml file, got: {input_path}")

    output_path = input_path.with_suffix(".json")

    tree = ET.parse(input_path)
    root = tree.getroot()

    function_measure = None

    for measure in root.findall("measure"):
        if measure.get("type") == "Function":
            function_measure = measure
            break

    if function_measure is None:
        raise ValueError("No <measure type=\"Function\"> section found in XML.")

    functions: list[dict[str, object]] = []

    for item in function_measure.findall("item"):
        values = [value.text for value in item.findall("value")]

        if len(values) < 3:
            continue

        parsed_name = parse_item_name(item.get("name", ""))

        functions.append(
            {
                "name": parsed_name["name"],
                "file": parsed_name["file"],
                "line": parsed_name["line"],
                "nloc": int(values[1]),
                "ccn": int(values[2]),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(functions, f, indent=2)

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())