"""Visualize the data-preprocessing / QC results into the results/ folder.

Figures produced:
  results/physiology_distributions.png  -- per-variable boxplots split by
      thermal stage, with individual samples overlaid so outliers are visible.
  results/protein_missingness.png       -- protein missingness profile with the
      drop threshold, and the low-variance profile with its threshold.
  results/protein_filtering_summary.png -- proteins kept vs dropped (by reason).

Run after preprocess.py (it reads the same artifacts).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from read_physiological_data import DATA_DIR, PROJECT_DIR
from preprocess import MAX_MISSING_FRAC, MIN_VARIANCE, PHYS_META

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Okabe-Ito colorblind-safe palette (validated, widely used for CVD safety).
C_NORMO = "#0072B2"   # blue  -> Normothermic (cool)
C_HYPER = "#D55E00"   # vermillion -> Hyperthermic (hot)
C_KEEP = "#0072B2"
C_DROP_MISS = "#D55E00"
C_DROP_VAR = "#E69F00"  # orange
C_THRESH = "#CC3311"    # threshold reference lines
INK = "#222222"
GRID = "#DDDDDD"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def _recessive(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_physiology_distributions(phys):
    phys_vars = [c for c in phys.columns if c not in PHYS_META]
    stages = ["Normothermic", "Hyperthermic"]
    colors = {"Normothermic": C_NORMO, "Hyperthermic": C_HYPER}

    ncol = 4
    nrow = int(np.ceil(len(phys_vars) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
    axes = axes.ravel()

    for ax, var in zip(axes, phys_vars):
        _recessive(ax)
        data = [phys.loc[phys["Thermal_Stage"] == s, var].values for s in stages]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, linewidth=1.4))
        for patch, s in zip(bp["boxes"], stages):
            patch.set_facecolor(colors[s])
            patch.set_alpha(0.30)
            patch.set_edgecolor(colors[s])
        # Overlay individual samples (jittered) so every point / outlier shows.
        for i, s in enumerate(stages):
            y = phys.loc[phys["Thermal_Stage"] == s, var].values
            jitter = np.linspace(-0.16, 0.16, len(y))
            ax.scatter(np.full(len(y), i) + jitter, y, s=22,
                       color=colors[s], edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(var, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normo", "Hyper"], fontsize=9)

    for ax in axes[len(phys_vars):]:
        ax.set_visible(False)

    # Single shared legend (identity is not color-alone: axis labels also name it).
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[s],
                          markeredgecolor="white", label=s) for s in stages]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Physiological variable distributions by thermal stage "
                 "(points = individual samples)", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "physiology_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_protein_missingness(miss_pct, variance):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # -- Missingness: discrete levels (0, 2.5, 5, 7.5, 10, 100 %) --
    _recessive(ax1)
    counts = miss_pct.round(1).value_counts().sort_index()
    bars = ax1.bar(range(len(counts)), counts.values, color=C_NORMO, alpha=0.85,
                   width=0.7, zorder=2)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels([f"{v:g}" for v in counts.index])
    ax1.set_xlabel("Missing values per protein (% of 40 samples)")
    ax1.set_ylabel("Number of proteins (log scale)")
    ax1.set_title("Protein missingness")
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                 va="bottom", fontsize=8)
    thr = MAX_MISSING_FRAC * 100
    ax1.axvline(np.interp(thr, counts.index, range(len(counts))),
                color=C_THRESH, linestyle="--", linewidth=1.4)
    ax1.text(0.97, 0.95, f"drop > {thr:g}% missing\n(removes {(miss_pct > thr).sum()})",
             transform=ax1.transAxes, ha="right", va="top", color=C_THRESH, fontsize=9)

    # -- Variance: histogram on log10 axis --
    _recessive(ax2)
    logv = np.log10(variance.dropna())
    ax2.hist(logv, bins=45, color=C_NORMO, alpha=0.85, zorder=2)
    ax2.axvline(np.log10(MIN_VARIANCE), color=C_THRESH, linestyle="--", linewidth=1.4)
    ax2.set_xlabel("log10(variance)  [NPX, log2 scale]")
    ax2.set_ylabel("Number of proteins")
    ax2.set_title("Protein variance")
    ax2.text(0.03, 0.95,
             f"drop variance < {MIN_VARIANCE}\n(removes {(variance < MIN_VARIANCE).sum()})",
             transform=ax2.transAxes, ha="left", va="top", color=C_THRESH, fontsize=9)

    fig.suptitle("Protein filters: missingness & variance", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "protein_missingness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_filtering_summary(n_total, dropped):
    n_miss = int((dropped["Reason"].str.startswith("missing")).sum())
    n_var = int((dropped["Reason"].str.startswith("variance")).sum())
    n_keep = n_total - n_miss - n_var

    labels = ["Kept", f"Dropped: missing > {MAX_MISSING_FRAC:.0%}",
              f"Dropped: variance < {MIN_VARIANCE}"]
    values = [n_keep, n_miss, n_var]
    colors = [C_KEEP, C_DROP_MISS, C_DROP_VAR]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    _recessive(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    bars = ax.barh(labels, values, color=colors, alpha=0.9, height=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of proteins (log scale)")
    ax.invert_yaxis()
    for b, v in zip(bars, values):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v}", va="center",
                ha="left", fontsize=10, fontweight="bold")
    ax.set_title(f"Protein filtering result  ({n_total} in → {n_keep} kept)",
                 fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "protein_filtering_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    merged = pd.read_csv(DATA_DIR / "Physiological_NPX_Merged.csv")
    dropped = pd.read_csv(DATA_DIR / "preprocess_dropped_proteins.csv")

    phys_cols = [c for c in merged.columns if c in phys.columns]
    protein_cols = [c for c in merged.columns if c not in phys_cols]
    miss_pct = merged[protein_cols].isna().mean() * 100
    variance = merged[protein_cols].var(numeric_only=True)

    outs = [
        plot_physiology_distributions(phys),
        plot_protein_missingness(miss_pct, variance),
        plot_filtering_summary(len(protein_cols), dropped),
    ]
    print("Saved figures:")
    for o in outs:
        print(" -", o)


if __name__ == "__main__":
    main()
