"""Corrected version of the protein-missingness panel.

The original panel (`protein_missingness.png`, left) is hard to read for three
reasons, all of them axis encoding rather than anything wrong with the data:

  1. bars are drawn at `range(len(counts))`, so the x-axis is CATEGORICAL --
     the 10% -> 100% step is drawn the same width as 0% -> 2.5%, which hides a
     36-sample gap;
  2. the y-axis is log, so 1075 vs 5 does not look like a 200-fold difference;
  3. the 50% threshold line is placed by interpolating into that categorical
     axis, so it lands in a position that has no numeric meaning.

The underlying distribution is simply discrete and strongly bimodal: with 40
samples a protein can only be missing in 0, 1, 2, 3, 4 ... samples, and in this
dataset NOTHING falls between 5 and 39 -- an Olink assay either works in nearly
every sample or fails outright (all-NaN, QC failure). This script redraws it on
a true numeric axis with an explicit break, and a linear count axis.

Output: results/preprocessing/protein_missingness_corrected.png
Run with any Python 3.8+ (pandas, numpy, matplotlib).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR, PROJECT_DIR

N_SAMPLES = 40
KEEP_WGCNA = 4        # WGCNA input keeps proteins missing in <= 4/40 (10%)
DROP_QC = 20          # differential-expression filter drops > 20/40
C_MAIN, C_FAIL, C_CUT = "#2E7EB8", "#C0392B", "#D55E00"


def main():
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    merged = pd.read_csv(DATA_DIR / "Physiological_NPX_Merged.csv")
    protein_cols = [c for c in merged.columns if c not in set(phys.columns)]
    n_missing = merged[protein_cols].isna().sum()          # 0..40 per protein

    counts = n_missing.value_counts().sort_index()
    lo = counts[counts.index <= 10]
    hi = counts[counts.index >= 30]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.4), sharey=True,
        gridspec_kw={"width_ratios": [6, 1], "wspace": 0.06})

    for ax, part, color in ((axL, lo, C_MAIN), (axR, hi, C_FAIL)):
        bars = ax.bar(part.index, part.values, width=0.62, color=color,
                      alpha=0.9, zorder=3)
        for b, v in zip(bars, part.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 18, f"{v:,}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold", color=color)
        ax.grid(axis="y", color="#EEEEEE", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axL.set_xlim(-0.7, 10.7)
    axL.set_xticks(range(0, 11))
    axR.set_xlim(38.6, 41.4)
    axR.set_xticks([40])
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)

    # axis-break marks
    kw = dict(transform=axL.transAxes, color="#444", clip_on=False, lw=1.2)
    axL.plot([1.003, 1.017], [-0.022, 0.022], **kw)
    kw["transform"] = axR.transAxes
    axR.plot([-0.10, -0.02], [-0.022, 0.022], **kw)

    axL.axvline(KEEP_WGCNA + 0.5, color=C_CUT, ls="--", lw=1.6, zorder=4)
    axL.text(KEEP_WGCNA + 0.72, counts.max() * 0.86,
             f"WGCNA input: keep ≤ {KEEP_WGCNA}/40 missing\n"
             f"({int(counts[counts.index <= KEEP_WGCNA].sum()):,} of "
             f"{len(protein_cols):,} proteins)",
             color=C_CUT, fontsize=9, va="top")

    axL.set_ylabel("Number of proteins")
    axL.set_ylim(0, counts.max() * 1.16)
    fig.text(0.5, -0.02, "Number of samples in which the protein is missing "
                         "(out of 40)", ha="center", fontsize=11)
    axL.set_title("Detected in nearly every sample", fontsize=10,
                  fontweight="bold", color=C_MAIN)
    axR.set_title("Assay\nfailed", fontsize=10, fontweight="bold", color=C_FAIL)

    axL.annotate("", xy=(10.6, counts.max() * 0.36), xytext=(5.4, counts.max() * 0.36),
                 arrowprops=dict(arrowstyle="<->", color="#888", lw=1.1))
    axL.text(8.0, counts.max() * 0.40, "no protein here\n(5–39 missing)",
             ha="center", fontsize=8.5, color="#777", style="italic")

    fig.suptitle("Protein detection is bimodal: an assay either works or it fails\n"
                 "linear counts, true numeric axis with an explicit break",
                 fontsize=12.5, fontweight="bold", y=1.06)
    out = PROJECT_DIR / "results" / "preprocessing" / "protein_missingness_corrected.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(counts.to_string())
    print(f"\ntotal proteins: {len(protein_cols):,}")
    print(f"missing in <= {KEEP_WGCNA}/40: {int(counts[counts.index <= KEEP_WGCNA].sum()):,}")
    print(f"missing in > {DROP_QC}/40 : {int(counts[counts.index > DROP_QC].sum()):,}")
    print("saved:", out)


if __name__ == "__main__":
    main()
