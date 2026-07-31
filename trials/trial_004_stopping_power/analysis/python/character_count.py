import os
import glob
import matplotlib.pyplot as plt
import re

SRC_DIR = "../src"
EXTENSIONS = (".cpp", ".hpp", ".cu", ".c", ".h")

def character_count_file(path):
    total_chars = 0
    comment_chars = 0
    code_chars = 0
    blank_chars = 0

    in_block_comment = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            # Count blank lines
            if not stripped:
                blank_chars += len(line)
                total_chars += len(line)
                continue

            # -----------------------------
            # Inside block comment
            # -----------------------------
            if in_block_comment:
                comment_chars += len(line)
                total_chars += len(line)

                if "*/" in line:
                    in_block_comment = False

                continue

            # -----------------------------
            # Full-line block comment
            # -----------------------------
            if stripped.startswith("/*"):
                comment_chars += len(line)
                total_chars += len(line)

                if "*/" not in line:
                    in_block_comment = True

                continue

            # -----------------------------
            # Single-line comment
            # -----------------------------
            if stripped.startswith("//"):
                comment_chars += len(line)
                total_chars += len(line)
                continue

            # -----------------------------
            # Inline block comment
            # -----------------------------
            if "/*" in line:
                before, after = line.split("/*", 1)

                code_chars += len(before)
                comment_chars += len(after)

                total_chars += len(line)

                if "*/" not in after:
                    in_block_comment = True

                continue

            # -----------------------------
            # Normal code
            # -----------------------------
            code_chars += len(line)
            total_chars += len(line)

    return {
        "total": total_chars,
        "blank": blank_chars,
        "comment": comment_chars,
        "code": code_chars
    }


def character_count():

    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(SRC_DIR, f"*{ext}")))

    results = {}

    for file in files:
        results[file] = character_count_file(file)

    # --- Print summary ---
    print("\nPer-file LOC:\n")
    for f, r in results.items():
        print(f"{f}")
        print(f"  code:    {r['code']}")
        print(f"  comment: {r['comment']}")
        print()

    # --- Prepare plot data ---
    files_sorted = sorted(results.keys())
    code_counts = [results[f]["code"] for f in files_sorted]

    # Shorten filenames for readability
    files_sorted = [
        os.path.splitext(os.path.basename(f))[0]
        for f in files_sorted
    ]

    plot_horizontal_bar(
        labels=files_sorted,
        values=code_counts,
        xlabel="Character Count (excluding comments)",
        title="Trial_004: Source Code Character Count",
        output_dir="analysis",
        output_file="complexity_characters_per_file.png",
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
            elif "parallel" in l:
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
    character_count()


if __name__ == "__main__":
    main()
