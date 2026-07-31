 
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Input files
# ----------------------------------------------------------------------

files = {
    "Add": "output_add.csv",
    "Divide": "output_divide.csv",
    "Exp": "output_exp.csv",
    "Log": "output_log.csv",
    "Multiply": "output_multiply.csv",
    "Power": "output_power.csv",
    "Sqrt": "output_sqrt.csv",
}

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

frames = []

for test, filename in files.items():
    data = pd.read_csv(
        filename,
        header=None,
        names=["Framework", "Mean", "Std"],
        sep=None,          # automatically detect comma/tab/etc.
        engine="python",
    )

    data.insert(0, "Test", test)

    frames.append(data)

df = pd.concat(frames, ignore_index=True)

# Keep only 32-bit implementations
df = df[df["Framework"].str.contains("_32_")].copy()

# Absolute error
df["Absolute Mean"] = df["Mean"].abs()

print(df)



fig, ax = plt.subplots(figsize=(12, 6))

for test in df["Test"].unique():
    subset = df[df["Test"] == test]

    ax.errorbar(
        subset["Framework"],
        subset["Absolute Mean"],
        yerr=subset["Std"],
        fmt="o",
        linestyle="none",
        capsize=4,
        label=test,
    )

ax.set_yscale("log")

ax.set_xlabel("Framework")
ax.set_ylabel("Absolute mean error")

ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", which="both", alpha=0.3)
ax.legend(title="Test")

fig.tight_layout()
plt.show()

