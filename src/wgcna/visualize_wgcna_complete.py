"""Visualize the COMPLETE-CASE WGCNA preprocessing (results/wgcna_complete/).

Parallel to visualize_wgcna.py but for the drop-any-missing workflow
(preprocess_wgcna_complete.py). Same 6-panel overview:
  1. filtering waterfall            2. per-protein missingness profile
  3. per-sample NPX distributions   4. PCA by participant
  5. PCA by exposure stage          6. sample dendrogram

Run after preprocess_wgcna_complete.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# shared ingestion modules live in src/data_pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from preprocess_wgcna import MERGED_PATH, _load_raw_long, below_lod_matrix

RESULTS_DIR = PROJECT_DIR / "results" / "wgcna" / "complete"
EXPR_PATH = DATA_DIR / "wgcna" / "complete" / "wgcna_expression.csv"

C_THRESH = "#CC3311"
GRID = "#DDDDDD"
EXPOSURE_COLORS = {"PR1": "#56B4E9", "PT1": "#E69F00",
                   "PR2": "#0072B2", "PT2": "#D55E00"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
})


def _recessive(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(EXPR_PATH, index_col=0)              # 40 x kept proteins
    merged = pd.read_csv(MERGED_PATH)
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    prot = [c for c in merged.columns if c not in phys.columns]

    sample_id = merged["Participant"].astype(str) + "-" + merged["Exposure"].astype(str)
    full = merged[prot].copy(); full.index = sample_id
    full = full[full.columns[full.isna().mean() < 1.0]]        # drop 5 assay-QC
    raw_long = _load_raw_long()
    blod = below_lod_matrix(raw_long, full.index, full.columns)
    miss_frac = (full.isna() | blod).mean(axis=0)

    n0, n1, n_keep = len(prot), full.shape[1], expr.shape[1]
    parts = [s.split("-")[0] for s in expr.index]
    exps = [s.split("-")[1] for s in expr.index]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # -- 1. filtering waterfall --
    ax = axes[0, 0]; _recessive(ax)
    steps, vals = ["merged", "assay-QC", "kept\n(complete\ncase)"], [n0, n1, n_keep]
    ax.bar(steps, vals, color=["#BBBBBB", "#0072B2", "#009E73"], alpha=0.9, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Proteins retained")
    ax.set_title(f"Protein filtering  ({n0} → {n_keep})")

    # -- 2. missingness distribution (kept = 0% missing only) --
    ax = axes[0, 1]; _recessive(ax)
    # Missingness is DISCRETE: with 40 samples only 41 values are possible
    # (0/40 .. 40/40). bins=40 aliases 41 levels into 40 bins, and floating
    # point (23/40*100 = 57.49999999999999) pushes one level into its
    # neighbour, leaving a phantom empty bin at 57.5%. Bin on the counts.
    ax.hist(miss_frac * 40, bins=np.arange(-0.5, 41.5, 1), color="#0072B2",
            alpha=0.85, zorder=2)
    ax.axvline(0, color=C_THRESH, linestyle="--", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Missing (NaN or < LOD) per protein  (samples of 40)")
    ax.set_ylabel("Proteins (log)")
    ax.set_title("Detection / missingness")
    ax.text(2, ax.get_ylim()[1] * 0.5,
            f"keep only 0% missing\n({n_keep} proteins)",
            color=C_THRESH, fontsize=8, va="top")

    # -- 3. per-sample NPX distributions --
    ax = axes[0, 2]; _recessive(ax)
    order = np.argsort(exps)
    data = [expr.iloc[i].values for i in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222", linewidth=1))
    for patch, i in zip(bp["boxes"], order):
        patch.set_facecolor(EXPOSURE_COLORS[exps[i]]); patch.set_alpha(0.6)
        patch.set_edgecolor(EXPOSURE_COLORS[exps[i]])
    ax.set_xticks([]); ax.set_xlabel("40 samples (grouped by exposure)")
    ax.set_ylabel("NPX (log2)")
    ax.set_title("Per-sample distributions (comparability)")

    # -- PCA --
    X = StandardScaler().fit_transform(expr.values)
    pcs = PCA(n_components=2).fit(X)
    xy = pcs.transform(X)
    v1, v2 = pcs.explained_variance_ratio_[:2] * 100

    ax = axes[1, 0]; _recessive(ax)
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(sorted(set(parts))):
        idx = [j for j, q in enumerate(parts) if q == p]
        ax.scatter(xy[idx, 0], xy[idx, 1], color=cmap(i % 10), s=40, label=p,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel(f"PC1 ({v1:.0f}%)"); ax.set_ylabel(f"PC2 ({v2:.0f}%)")
    ax.set_title("PCA — coloured by participant")
    ax.legend(fontsize=6, ncol=2, frameon=False, title="participant")

    ax = axes[1, 1]; _recessive(ax)
    for e, c in EXPOSURE_COLORS.items():
        idx = [j for j, q in enumerate(exps) if q == e]
        ax.scatter(xy[idx, 0], xy[idx, 1], color=c, s=40, label=e,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel(f"PC1 ({v1:.0f}%)"); ax.set_ylabel(f"PC2 ({v2:.0f}%)")
    ax.set_title("PCA — coloured by exposure stage")
    ax.legend(fontsize=8, frameon=False, title="exposure")

    ax = axes[1, 2]
    link = linkage(expr.values, method="average", metric="euclidean")
    dendrogram(link, labels=list(expr.index), ax=ax, leaf_font_size=6,
               color_threshold=0, above_threshold_color="#888888")
    ax.set_title("Sample clustering")
    ax.set_ylabel("Euclidean distance")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(f"WGCNA preprocessing overview — COMPLETE CASE "
                 f"({n_keep} proteins × 40 samples, no imputation)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = RESULTS_DIR / "wgcna_preprocessing.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved figure:", out)


if __name__ == "__main__":
    main()
