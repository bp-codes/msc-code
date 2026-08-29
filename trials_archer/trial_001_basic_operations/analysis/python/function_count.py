import os
import glob
import matplotlib.pyplot as plt
import re

SRC_DIR = "../src"
EXTENSIONS = (".cpp", ".hpp", ".cu", ".c", ".h")


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


def function_count():

    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(SRC_DIR, f"*{ext}")))

    results = {}

    for file in files:
        results[file] = function_count_file(file)

    # --- Print summary ---
    print("\nPer-file LOC:\n")
    for r in results.items():
        print(f"  code:    {r}")
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
        xlabel="Function Count",
        title="Trial_001: Source Code Function Count",
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

        # --- Colour logic ---
        if use_greyscale:
            colour = "0.6"
        else:
            if ("precise" in l):
                colour = "#f4a3a3"
            elif ("cuda" in l or "sycl" in l or "opencl" in l or "hip" in l):
                colour = "#a9d6a5"
            elif ("parallel" in l or "openmp" in l):
                colour = "#a8c9f0"
            elif "serial" in l:
                colour = "#f6c28b"
            else:
                colour = "grey"

        colours.append(colour)

        # --- Hatch logic ---
        if "32" in l:
            hatch = "///"   # 'xx', '...', '\\\\'
        else:
            hatch = None

        hatches.append(hatch)

    # Create figure
    plt.figure(figsize=(width, height))

    bars = plt.barh(labels_sorted, values_sorted,
                    color=colours,
                    edgecolor="black")
    plt.bar_label(bars, fmt="%d", padding=3)
    plt.margins(x=0.2) 

    # Apply hatches individually
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    plt.xlabel(xlabel)
    plt.title(title)

    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot: {plot_path}")


def main():
    function_count()


if __name__ == "__main__":
    main()
