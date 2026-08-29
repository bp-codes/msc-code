import os
import glob
import matplotlib.pyplot as plt
import re

SRC_DIR = "../src"
EXTENSIONS = (".cpp", ".hpp", ".cu", ".c", ".h")


def word_count_file(path):
    total_words = 0
    comment_words = 0
    code_words = 0

    in_block_comment = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            # -----------------------------
            # Inside block comment
            # -----------------------------
            if in_block_comment:
                words = re.findall(r"\w+", stripped)
                comment_words += len(words)

                if "*/" in stripped:
                    in_block_comment = False

                continue

            # -----------------------------
            # Full-line block comment
            # -----------------------------
            if stripped.startswith("/*"):
                words = re.findall(r"\w+", stripped)
                comment_words += len(words)

                if "*/" not in stripped:
                    in_block_comment = True

                continue

            # -----------------------------
            # Single-line comment
            # -----------------------------
            if stripped.startswith("//"):
                words = re.findall(r"\w+", stripped)
                comment_words += len(words)
                continue

            # -----------------------------
            # Inline block comment
            # -----------------------------
            if "/*" in stripped:
                code_part = stripped.split("/*")[0]
                comment_part = stripped.split("/*", 1)[1]

                code_words += len(re.findall(r"\w+", code_part))
                comment_words += len(re.findall(r"\w+", comment_part))

                if "*/" not in stripped:
                    in_block_comment = True

                continue

            # -----------------------------
            # Normal code
            # -----------------------------
            code_words += len(re.findall(r"\w+", stripped))

    total_words = code_words + comment_words

    return {
        "total": total_words,
        "comment": comment_words,
        "code": code_words
    }


def word_count():

    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(SRC_DIR, f"*{ext}")))

    results = {}

    for file in files:
        results[file] = word_count_file(file)

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
        xlabel="Word Count (excluding comments)",
        title="Trial_001: Source Code Word Count",
        output_dir="analysis",
        output_file="complexity_words_per_file.png",
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
    word_count()


if __name__ == "__main__":
    main()
