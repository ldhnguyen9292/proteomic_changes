"""Eigengene relationships within a network, and the merge-threshold check.

A module eigengene is a vector over the samples of ONE contrast, so eigengenes are
only comparable within the same network. M1 and M3 both come from PT1/PT2 and can
be compared directly; M2 comes from PR2/PT2, which shares only the 10 PT2 samples,
so it is plotted separately and NOT placed in the same matrix.

The point of the figure is the merge threshold. WGCNA merges two modules when their
eigengenes correlate above 1 - MEDissThres (0.8 here). M1 and M3 sit just under it,
so the three-module result depends on that setting -- the figure marks how close.

Outputs (results/wgcna/):
  eigengene_network.png
  eigengene_correlations.csv

Run with Python 3.9 (PyWGCNA).
"""

import contextlib
import io
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
import model_development as md

CASE = DATA_DIR / "wgcna" / "impute"
OUT = PROJECT_DIR / "results" / "wgcna"
MERGE_AT = 0.8          # 1 - MEDissThres; above this WGCNA merges two modules
LABEL = {"dimgrey": "M1", "lightgrey": "M3", "black": "M2"}
TRAIT = "LSR (mg/min/cm2)"


def _lab(net, mod):
    tag = LABEL.get(mod, "")
    sel = (net, mod) in (("PT1/PT2", "dimgrey"), ("PT1/PT2", "lightgrey"),
                         ("PR2/PT2", "black"))
    return f"{mod}\n({tag})" if sel else mod


def main():
    expr = pd.read_csv(CASE / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    nets = {}
    pkl = Path("/tmp/pt1pt2_obj.pkl")
    if pkl.exists():
        nets["PT1/PT2"] = pickle.load(open(pkl, "rb"))
    for name, codes in (("PT1/PT2", ["PT1", "PT2"]), ("PR2/PT2", ["PR2", "PT2"])):
        if name in nets:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            nets[name] = md.build(expr, tr, pd.Series(ex.isin(codes), index=tr.index),
                                  name.replace("/", "_"))

    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.34, 1], width_ratios=[1, 1, 0.9],
                          hspace=0.06, wspace=0.34, top=0.80, bottom=0.21,
                          left=0.055, right=0.965)

    rows = []
    for j, (name, obj) in enumerate(nets.items()):
        ME = obj.MEs.copy()
        ME.columns = [c[2:] for c in ME.columns]
        C = ME.corr()
        for a in C.index:
            for b in C.columns:
                if a < b:
                    rows.append(dict(network=name, module_a=a, module_b=b,
                                     r=round(float(C.loc[a, b]), 3),
                                     merged=bool(C.loc[a, b] > MERGE_AT)))

        # dendrogram on 1 - r, the quantity WGCNA actually thresholds
        D = (1 - C).clip(lower=0)
        np.fill_diagonal(D.values, 0)
        link = linkage(squareform(D.values, checks=False), "average")
        axd = fig.add_subplot(gs[0, j])
        dn = dendrogram(link, labels=list(C.columns), ax=axd, color_threshold=0,
                        above_threshold_color="#4A4A4A", no_labels=True)
        axd.axhline(1 - MERGE_AT, color="#C0392B", ls="--", lw=1.3)
        axd.text(0.985, 1 - MERGE_AT, f"merge below {1-MERGE_AT:.1f}", fontsize=7.5,
                 color="#C0392B", ha="right", va="center", zorder=6,
                 transform=axd.get_yaxis_transform(),
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
        axd.set_ylim(0, max(link[:, 2].max() * 1.15, 0.35))
        axd.set_xticks([])
        axd.set_ylabel("1 − r", fontsize=8)
        axd.tick_params(labelsize=7)
        for s in ("top", "right"):
            axd.spines[s].set_visible(False)
        axd.set_title(f"{name}   ({ME.shape[0]} samples)", fontsize=11.5,
                      fontweight="bold")

        order = [C.columns[i] for i in dn["leaves"]]
        C = C.loc[order, order]
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_lab(name, m) for m in order], fontsize=8.5, rotation=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_lab(name, m) for m in order], fontsize=8.5)
        for a in range(len(order)):
            for b in range(len(order)):
                v = C.values[a, b]
                near = (a != b) and v > MERGE_AT - 0.1
                ax.text(b, a, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9.5 if near else 8.5,
                        fontweight="bold" if near else "normal",
                        color="white" if abs(v) > 0.55 else "#222")
                if near:
                    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False,
                                               edgecolor="#C0392B", lw=2.2))
        ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "eigengene_correlations.csv", index=False)

    # ---- notes panel ----
    axn = fig.add_subplot(gs[:, 2])
    axn.axis("off")
    m13 = tab[(tab.network == "PT1/PT2") &
              (tab.module_a.isin(["dimgrey", "lightgrey"])) &
              (tab.module_b.isin(["dimgrey", "lightgrey"]))].r.iloc[0]
    lines = [
        ("Why two panels, not one", 11.5, True, "#1F4E79"),
        ("An eigengene is a vector over the samples", 9.5, False, "#333"),
        ("of one contrast. PT1/PT2 and PR2/PT2 share", 9.5, False, "#333"),
        ("only the 10 PT2 samples, so eigengenes from", 9.5, False, "#333"),
        ("the two cannot go in one matrix. M2 is shown", 9.5, False, "#333"),
        ("separately for that reason.", 9.5, False, "#333"),
        ("", 6, False, "#333"),
        ("The merge threshold", 11.5, True, "#1F4E79"),
        (f"M1 vs M3:  r = {m13:+.3f}", 12, True, "#C0392B"),
        ("WGCNA merges two modules when their", 9.5, False, "#333"),
        ("eigengenes correlate above 0.80. M1 and M3", 9.5, False, "#333"),
        (f"sit {MERGE_AT - m13:.3f} below that line.", 9.5, False, "#333"),
        ("", 6, False, "#333"),
        ("So the three-module result depends on", 9.5, False, "#444"),
        ("MEDissThres = 0.2. At 0.3, M1 and M3", 9.5, False, "#444"),
        ("would have merged and there would be", 9.5, False, "#444"),
        ("two modules, not three.", 9.5, False, "#444"),
        ("", 6, False, "#333"),
        ("Red boxes mark pairs within 0.1 of the", 8.8, False, "#777"),
        ("merge threshold.", 8.8, False, "#777"),
    ]
    y = 0.985
    for t, sz, bold, col in lines:
        axn.text(0, y, t, fontsize=sz, fontweight="bold" if bold else "normal",
                 color=col, va="top", transform=axn.transAxes)
        y -= 0.047 if t else 0.022

    cax = fig.add_axes([0.055, 0.055, 0.40, 0.016])
    fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.set_xlabel("eigengene correlation (r)", fontsize=8.5)
    cax.tick_params(labelsize=8)

    fig.suptitle("Module eigengene relationships — and how close M1 and M3 came "
                 "to being one module", fontsize=13.5, fontweight="bold", y=0.95)
    f = OUT / "eigengene_network.png"
    fig.savefig(f, bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(tab.to_string(index=False))
    print(f"\nM1 vs M3 = {m13:+.3f}  (merge threshold {MERGE_AT}, "
          f"gap {MERGE_AT - m13:.3f})")
    print("saved:", OUT / "eigengene_correlations.csv")
    print("saved:", f)


if __name__ == "__main__":
    main()
