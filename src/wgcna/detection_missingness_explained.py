"""Annotated version of the 'Detection / missingness' panel of wgcna_preprocessing.png.

The WGCNA panel counts a value as MISSING if it was QC-excluded (NaN) **or below
the limit of detection (NPX < LOD)**. That is a different definition from the
QC-only missingness in results/preprocessing/protein_missingness.png, which is
why the two figures look nothing alike, and it is the whole reason this one is
U-shaped.

With 40 samples only 41 missingness values are possible (0/40 .. 40/40). The
original panel used `bins=40`, which aliases 41 discrete levels into 40 bins;
floating point (23/40*100 = 57.49999999999999) then pushes one level into its
neighbour and leaves a phantom empty bin at 57.5%. Every level is in fact
occupied. This version bins on the counts, so one bar = one level.

Output: results/wgcna/detection_missingness_explained.png
Run with Python 3.8+ (needs olink_raw for LOD).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from preprocess_wgcna import (MERGED_PATH, MAX_MISSING_FRAC, _load_raw_long,
                              below_lod_matrix)

KEEP = int(MAX_MISSING_FRAC * 40)          # 4 of 40
C_KEEP, C_MID, C_NEVER, C_CUT = "#0072B2", "#9C9C9C", "#C0392B", "#D55E00"


def main():
    merged = pd.read_csv(MERGED_PATH)
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    prot = [c for c in merged.columns if c not in set(phys.columns)]
    sid = merged["Participant"].astype(str) + "-" + merged["Exposure"].astype(str)
    full = merged[prot].copy()
    full.index = sid
    full = full[full.columns[full.isna().mean() < 1.0]]        # drop 5 assay-QC
    blod = below_lod_matrix(_load_raw_long(), full.index, full.columns)
    n_missing = (full.isna() | blod).sum(axis=0)

    counts = n_missing.value_counts().reindex(range(41), fill_value=0)
    colors = [C_KEEP if k <= KEEP else (C_NEVER if k == 40 else C_MID)
              for k in counts.index]

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar(counts.index, counts.values, width=0.82, color=colors, alpha=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(8, counts.max() * 9)
    ax.set_xlim(-0.8, 40.8)
    ax.set_xticks(range(0, 41, 2))
    ax.set_xlabel("Number of samples in which the protein is missing  "
                  "(NaN or below LOD, out of 40)", fontsize=10.5)
    ax.set_ylabel("Number of proteins  (log scale)", fontsize=10.5)
    ax.grid(axis="y", color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.axvline(KEEP + 0.5, color=C_CUT, ls="--", lw=1.7, zorder=5)

    kept = int(counts[counts.index <= KEEP].sum())
    mid = int(counts[(counts.index > KEEP) & (counts.index < 40)].sum())
    never = int(counts[40])

    def band(x0, x1, color, title, lines):
        ax.axvspan(x0, x1, color=color, alpha=0.055, zorder=0)
        ax.text((x0 + x1) / 2, counts.max() * 5.2, title, ha="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text((x0 + x1) / 2, counts.max() * 3.1, lines, ha="center",
                fontsize=8.6, color="#444", va="top")

    band(-0.8, KEEP + 0.5, C_KEEP, "Well detected → KEPT",
         f"{kept:,} proteins\nabove LOD in ≥ 36 of 40 samples")
    band(KEEP + 0.5, 39.5, C_MID, "Sitting on the detection limit → dropped",
         f"{mid:,} proteins\ndetected in some samples, not others")
    band(39.5, 40.8, C_NEVER, "Never detected",
         f"{never:,} proteins\nbelow LOD in all 40")

    ax.annotate(f"{int(counts[0]):,}", xy=(0, counts[0]), xytext=(0, counts[0] * 1.35),
                ha="center", fontsize=9, fontweight="bold", color=C_KEEP)
    ax.annotate(f"{never:,}", xy=(40, never), xytext=(40, never * 1.35),
                ha="center", fontsize=9, fontweight="bold", color=C_NEVER)
    ax.text(KEEP + 0.9, 11, f"cut-off: > {KEEP}/40 missing\n"
                            f"drops {mid + never:,} proteins",
            color=C_CUT, fontsize=9, va="bottom")

    fig.suptitle("Detection / missingness — why it is U-shaped\n"
                 "'missing' here = QC-excluded OR below the limit of detection; "
                 "one bar = one of the 41 possible values",
                 fontsize=12.5, fontweight="bold", y=1.04)
    out = PROJECT_DIR / "results" / "wgcna" / "detection_missingness_explained.png"
    fig.savefig(out, bbox_inches="tight", dpi=190)
    plt.close(fig)

    print(f"kept (≤{KEEP}/40): {kept:,}   partial ({KEEP+1}-39): {mid:,}   "
          f"never (40/40): {never:,}   total dropped: {mid + never:,}")
    print("levels with zero proteins:", [k for k in range(41) if counts[k] == 0])
    print("saved:", out)


if __name__ == "__main__":
    main()
