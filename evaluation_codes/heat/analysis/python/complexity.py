import os
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT_DIR = "complexity"


def plot_style(string, use_greyscale=False):
    # --- Colour logic ---
    if use_greyscale:
        colour = "0.6"
    else:
        if "precise" in string:
            colour = "#f4a3a3"
        elif ("cuda" in string or "sycl" in string or "opencl" in string or "hip" in string):
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


def parse_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    # Find file-level metrics
    for measure in root.findall(".//measure[@type='File']"):
        for item in measure.findall("item"):
            name = item.get("name")

            values = [int(v.text) for v in item.findall("value")]

            if len(values) >= 4:
                ncss = values[1]        # total NLOC-like metric
                func_count = values[3]  # number of functions

                results.append({
                    "file": name,
                    "nloc": ncss,
                    "functions": func_count
                })

    return results


def main():
    xml_files = glob.glob(os.path.join(INPUT_DIR, "*.xml"))

    print(xml_files)
    os.makedirs("analysis", exist_ok=True)

    all_results = []
    for xml_file in xml_files:
        try:
            all_results.extend(parse_file(xml_file))
        except Exception as e:
            print(f"Failed to parse {xml_file}: {e}")

    if not all_results:
        print("No data found.")
        return

    # Aggregate (in case duplicates appear)
    nloc_per_file = defaultdict(int)
    func_per_file = defaultdict(int)

    for r in all_results:
        nloc_per_file[r["file"]] += r["nloc"]
        func_per_file[r["file"]] += r["functions"]

    files = sorted(nloc_per_file.keys())

    nloc_values = [nloc_per_file[f] for f in files]
    func_values = [func_per_file[f] for f in files]

    avg_cc_per_file = {}

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for measure in root.findall(".//measure[@type='Function']"):
            cc_sum = 0
            count = 0
            file_name = None

            for item in measure.findall("item"):
                values = [int(v.text) for v in item.findall("value")]
                if len(values) >= 3:
                    cc_sum += values[2]
                    count += 1

                    # extract file from name string
                    file_name = item.get("name").split(" at ")[-1].split(":")[0]

            if count > 0 and file_name:
                avg_cc_per_file[file_name] = cc_sum / count

    files = list(avg_cc_per_file.keys())
    vals = list(avg_cc_per_file.values())

    # Shorten filenames for readability
    files = [
        os.path.splitext(os.path.basename(f))[0]
        for f in files
    ]


    plot_horizontal_bar(
        labels=files,
        values=vals,
        xlabel="Total number of lines of code",
        title="Total NLOC (NCSS) per file",
        output_dir="analysis",
        output_file="complexity_nloc_per_file.png",
        width=8,
        height=6,
        use_greyscale=False  # or False
    )


    plot_horizontal_bar(
        labels=files,
        values=vals,
        xlabel="Function count",
        title="Function count per file",
        output_dir="analysis",
        output_file="complexity_function_count_per_file.png",
        width=8,
        height=6,
        use_greyscale=False  # or False
    )


    plot_horizontal_bar(
        labels=files,
        values=vals,
        xlabel="Average Cyclomatic Complexity)",
        title="Average Cyclomatic Complexity per File",
        output_dir="analysis",
        output_file="complexity_avg_cc_per_file.png",
        width=8,
        height=6,
        use_greyscale=False  # or False
    )



def plot_horizontal_bar(labels,
                        values,
                        xlabel,
                        title,
                        output_dir,
                        output_file,
                        width=8,
                        height=6,
                        use_greyscale=False):

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, output_file)

    # Sort descending
    pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels_sorted, values_sorted = zip(*pairs)

    # --- Build colours + hatches ---
    colours = []
    hatches = []

    for label in labels_sorted:
        l = label.lower()
        
        colour, hatch = plot_style(l)
        colours.append(colour)
        hatches.append(hatch)

    # Create figure
    plt.figure(figsize=(width, height))

    bars = plt.barh(labels_sorted, values_sorted,
                    color=colours,
                    edgecolor="black")

    # Apply hatches individually
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    plt.bar_label(bars, fmt="%.1f", padding=3)
    plt.margins(x=0.2) 
    plt.xlabel(xlabel)
    plt.title(title)

    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
