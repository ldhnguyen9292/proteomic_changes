"""How WGCNA separates proteins into modules — explanatory figure.

Three panels, following the actual steps PyWGCNA runs on the PT1/PT2 network:

  A. Soft-threshold selection. Correlations are raised to a power beta so that the
     network approximates scale-free topology; beta is the smallest power reaching
     R^2 > 0.8.
  B. Protein dendrogram + module assignment. Proteins are clustered on
     1 - TOM (topological overlap: two proteins are close if they are correlated
     AND share neighbours). Dynamic Tree Cut carves branches into modules
     (minModuleSize=30); modules whose eigengenes correlate > 0.8 are then merged.
     The two colour strips show before and after merging.
  C. Resulting modules and their sizes, with the M1/M2/M3 labels used in the
     report.

Colour names ('dimgrey', 'black', ...) are arbitrary labels PyWGCNA assigns by
module size rank -- they carry no biological meaning and can be renamed freely.

Usage (Python 3.9 with PyWGCNA):
  python module_detection_figure.py            # builds PT1/PT2, then plots
Output: results/wgcna/module_detection_explained.png
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
import model_development as md

NETWORK, CODES = "PT1/PT2", ["PT1", "PT2"]
R2_CUT = 0.8
# report labels for the modules that pass all three criteria
RENAME = {("PT1/PT2", "dimgrey"): "M1", ("PR2/PT2", "black"): "M2",
          ("PT1/PT2", "lightgrey"): "M3"}
# distinct, colour-blind-safe fills for the strips (the WGCNA names are greys)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9",
           "#F0E442", "#999999"]


def build():
    cdir = DATA_DIR / "wgcna" / "impute"
    expr = pd.read_csv(cdir / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(cdir / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]
    return md.build(expr, tr, pd.Series(ex.isin(CODES), index=tr.index),
                    NETWORK.replace("/", "_"))


def _strip(ax, order, labels, cmap, title):
    """Draw one horizontal colour strip in dendrogram-leaf order.

    `labels` is positional (one entry per protein, in datExpr.var order, which is
    the order the linkage in geneTree was built over).
    """
    arr = np.array([cmap[labels[i]] for i in order])[None, :, :]
    ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([0])
    ax.set_yticklabels([title], fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    obj = build()
    var = obj.datExpr.var
    final = var["moduleColors"].to_numpy()
    dyn = var["dynamicColors"].to_numpy()      # pre-merge assignment
    merged_any = not np.array_equal(dyn, final)
    link = np.asarray(obj.geneTree, dtype=float)

    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.55, 0.95],
                          width_ratios=[1, 1], hspace=0.55, wspace=0.22)

    # ---------- A. soft-threshold ----------
    axA = fig.add_subplot(gs[0, 0])
    sft = obj.sft
    axA.plot(sft["Power"], sft["SFT.R.sq"], "o-", color="#0072B2", lw=1.8, ms=5)
    axA.axhline(R2_CUT, color="#C0392B", ls="--", lw=1.4)
    axA.axvline(obj.power, color="#2E7D32", ls=":", lw=1.8)
    axA.annotate(f"chosen β = {obj.power}\nR² = {float(sft.loc[sft['Power'] == obj.power, 'SFT.R.sq'].iloc[0]):.3f}",
                 xy=(obj.power, float(sft.loc[sft["Power"] == obj.power, "SFT.R.sq"].iloc[0])),
                 xytext=(obj.power - 8.5, 0.42), fontsize=9, fontweight="bold",
                 color="#2E7D32",
                 arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2))
    axA.text(1, R2_CUT + 0.02, "scale-free fit R² > 0.8", color="#C0392B", fontsize=8.5)
    axA.set_xlabel("soft-threshold power  β")
    axA.set_ylabel("scale-free fit  R²")
    axA.set_title("A. Choose β so the network is scale-free", fontsize=10.5,
                  fontweight="bold", loc="left")
    axA.set_ylim(0, 1.02)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.grid(color="#EEEEEE")
    axA.set_axisbelow(True)

    # ---------- A2. the pipeline in words ----------
    axT = fig.add_subplot(gs[0, 1])
    axT.axis("off")
    steps = [
        ("1", "correlate every protein pair", "across the 20 samples of this contrast"),
        ("2", f"adjacency = |cor|^β  (β = {obj.power})", "signed hybrid: negative correlations → 0"),
        ("3", "TOM = topological overlap", "close = correlated AND sharing neighbours"),
        ("4", "cluster proteins on 1 − TOM", "average-linkage hierarchical tree (panel B)"),
        ("5", f"Dynamic Tree Cut, min size {obj.minModuleSize}", "each surviving branch becomes a module"),
        ("6", "merge modules with ME cor > 0.8", "eigengene distance < 0.2 → one module"),
    ]
    axT.text(0, 1.0, "How a module is defined", fontsize=10.5, fontweight="bold",
             va="top", transform=axT.transAxes)
    for i, (n, head, sub) in enumerate(steps):
        y = 0.85 - i * 0.152
        axT.text(0.0, y, n, fontsize=9, fontweight="bold", color="white", va="center",
                 transform=axT.transAxes,
                 bbox=dict(boxstyle="circle,pad=0.28", fc="#0072B2", ec="none"))
        axT.text(0.062, y + 0.028, head, fontsize=9.2, fontweight="bold", va="center",
                 transform=axT.transAxes)
        axT.text(0.062, y - 0.038, sub, fontsize=8.2, color="#555", va="center",
                 transform=axT.transAxes)

    # ---------- B. dendrogram + strips ----------
    axB = fig.add_subplot(gs[1, :])
    with plt.rc_context({"lines.linewidth": 0.35}):
        dn = dendrogram(link, no_labels=True, color_threshold=0,
                        above_threshold_color="#4A4A4A", ax=axB)
    order = dn["leaves"]
    axB.set_xticks([])
    # zoom to where the merges actually happen; near-1.0 heights otherwise
    # compress the whole structure into a solid block
    axB.set_ylim(float(link[:, 2].min()) * 0.985, 1.003)
    axB.set_ylabel("1 − TOM", fontsize=9)
    axB.set_title(f"B. Protein dendrogram — {NETWORK} network "
                  f"({len(final):,} proteins), cut into modules",
                  fontsize=10.5, fontweight="bold", loc="left")
    for s in ("top", "right", "bottom"):
        axB.spines[s].set_visible(False)

    dyn_lv = list(pd.unique(dyn))
    fin_lv = list(pd.unique(final))
    cmap_dyn = {v: matplotlib.colors.to_rgb(PALETTE[i % len(PALETTE)])
                for i, v in enumerate(dyn_lv)}
    cmap_fin = {v: matplotlib.colors.to_rgb(PALETTE[i % len(PALETTE)])
                for i, v in enumerate(fin_lv)}

    pos = axB.get_position()
    h = 0.026
    if merged_any:
        ax1 = fig.add_axes([pos.x0, pos.y0 - h - 0.012, pos.width, h])
        _strip(ax1, order, dyn, cmap_dyn, f"Dynamic Tree Cut\n({len(dyn_lv)} modules)")
        ax2 = fig.add_axes([pos.x0, pos.y0 - 2 * h - 0.022, pos.width, h])
        _strip(ax2, order, final, cmap_fin, f"After merging\n({len(fin_lv)} modules)")
    else:
        # For this network no two eigengenes correlated > 0.8, so the merge step
        # changed nothing -- showing a "before/after" pair would be misleading.
        ax1 = fig.add_axes([pos.x0, pos.y0 - h - 0.012, pos.width, h])
        _strip(ax1, order, final, cmap_fin, f"Modules\n({len(fin_lv)} found)")
        fig.text(pos.x0, pos.y0 - 2 * h - 0.002,
                 "Step 6 (merging) changed nothing here — no two module eigengenes "
                 "correlated above 0.8.   Colour blocks are not perfectly "
                 "contiguous because the Dynamic Tree Cut PAM stage assigns "
                 "leftover proteins to the nearest module, not strictly by branch.",
                 fontsize=8.0, color="#777", style="italic")

    # ---------- C. module table ----------
    axC = fig.add_subplot(gs[2, :])
    axC.axis("off")
    sizes = pd.Series(final).value_counts()
    rows = []
    for m in sizes.index:
        rows.append([RENAME.get((NETWORK, m), "—"), m, f"{sizes[m]:,}"])
    tbl = axC.table(cellText=rows,
                    colLabels=["report label", "WGCNA colour name", "proteins"],
                    cellLoc="center", loc="upper center",
                    colWidths=[0.16, 0.26, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.45)
    for j in range(3):
        tbl[0, j].set_facecolor("#0072B2")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, m in enumerate(sizes.index, start=1):
        tbl[i, 1].set_facecolor(matplotlib.colors.to_hex(cmap_fin[m]) + "40")
        if RENAME.get((NETWORK, m), "—") != "—":
            tbl[i, 0].set_text_props(fontweight="bold", color="#2E7D32")
    axC.set_title("C. Resulting modules — the colour name is an arbitrary label "
                  "assigned by size rank, and can be renamed",
                  fontsize=10.5, fontweight="bold", loc="left", y=0.98)

    fig.suptitle("How WGCNA separates proteins into modules",
                 fontsize=14, fontweight="bold", y=0.965)
    out = PROJECT_DIR / "results" / "wgcna" / "module_detection_explained.png"
    fig.savefig(out, bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(f"β = {obj.power}, minModuleSize = {obj.minModuleSize}, "
          f"MEDissThres = {obj.MEDissThres}")
    print(f"dynamic tree cut -> {len(dyn_lv)} modules; after merging -> {len(fin_lv)}")
    print(sizes.to_string())
    print("saved:", out)


if __name__ == "__main__":
    main()
