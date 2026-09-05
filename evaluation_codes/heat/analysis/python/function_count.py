import os
import glob
import matplotlib.pyplot as plt
import re

FRAMEWORKS = ["cuda", "cuda_32", "opencl", "opencl_32", "hip", "hip_32", "openmp", "openmp_32", "serial", "serial_32", "sycl", "sycl_32"]
SRC_DIR = "src"
INCLUDE_DIR = "include/heat"
EXTENSIONS = (".cpp", ".hpp", ".cu", ".c", ".h")


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


def function_count_file(path):

    with open(path, "r", errors="ignore") as f:
        text = f.read()

    # ----------------------------------------
    # Remove comments
    # ----------------------------------------
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # ----------------------------------------
    # Remove string literals
    # Helps avoid matching inside raw strings
    # ----------------------------------------
    text = re.sub(
        r'R"([^()]*)\(.*?\)\1"',
        '""',
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r'"(?:\\.|[^"\\])*"',
        '""',
        text
    )

    # ----------------------------------------
    # Function regex
    # ----------------------------------------
    pattern = re.compile(
        r"""
        ^\s*

        # Optional template
        (?:template\s*<[^;{]+>\s*)?

        # Optional attributes
        (?:\[\[.*?\]\]\s*)*

        # Optional qualifiers
        (?:
            inline|
            static|
            constexpr|
            virtual|
            explicit|
            friend|
            extern|
            __kernel
        )?\s*

        # Return type / qualifiers
        [\w:\<\>\~\*&\s]+?

        \s+

        # Function name
        ([A-Za-z_]\w*)

        \s*

        # Arguments
        \(
            [^;{}]*?
        \)

        # Optional qualifiers
        (?:\s+(?:const|noexcept|override|final))*

        \s*

        # Optional trailing return type
        (?:->\s*[\w:<>\*&]+)?

        \s*

        # Function body start
        \{
        """,
        re.MULTILINE | re.VERBOSE
    )

    excluded = {
        "if",
        "for",
        "while",
        "switch",
        "catch"
    }

    matches = []

    for m in pattern.finditer(text):

        name = m.group(1)

        if name in excluded:
            continue

        matches.append(name)

    return len(matches)


def main():

    results = {}

    for framework in FRAMEWORKS:
        files = []
        results[framework] = 0
        for ext in EXTENSIONS:
            search_path = os.path.join("../" + framework, os.path.join(SRC_DIR, f"*{ext}"))
            files.extend(glob.glob(search_path))
            search_path = os.path.join("../" + framework, os.path.join(INCLUDE_DIR, f"*{ext}"))
            files.extend(glob.glob(search_path))

        for file in files:
            print(file)
            file_loc = function_count_file(file)
            results[framework] += file_loc

    # --- Print summary ---
    print("\nPer-file Functions:\n")
    for f, r in results.items():
        print(f"{f}")
        print(f"  functions:    {r}")
        print()

    # --- Prepare plot data ---
    files_sorted = sorted(results.keys())
    code_counts = [results[f] for f in files_sorted]

    # Shorten filenames for readability
    files_sorted = [
        os.path.splitext(os.path.basename(f))[0]
        for f in files_sorted
    ]

    plot_horizontal_bar(
        labels=files_sorted,
        values=code_counts,
        xlabel="Functions",
        title="Heat2D: Source Code Function Count",
        output_dir="analysis",
        output_file="complexity_functions_per_file.png",
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

    plt.bar_label(bars, fmt="%d", padding=3)
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
