"""Visualize the differential-expression recreation into results/.

Reads the per-contrast tables written by differential_expression.py (so the
statistics are computed once, in one place) and produces:

  results/de_volcano.png   -- volcano plot per contrast; "changed" proteins
      (BH-FDR < 0.05) coloured by direction, abstract marker genes labelled.
  results/de_overlap.png   -- overlap of the pre- vs post-acclimation heat
      responses, showing the proteins changed only after acclimation.
  results/de_summary.png   -- recreated counts next to the abstract's figures.

Run after differential_expression.py.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from read_physiological_data import PROJECT_DIR
from differential_expression import (CONTRASTS, MARKERS, FDR_CHANGED, FDR_STRICT,
                                      DATA_DIR)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

C_UP = "#D55E00"     # vermillion -> increased in concentration
C_DOWN = "#0072B2"   # blue       -> decreased
C_NS = "#BBBBBB"     # grey       -> not changed
C_THRESH = "#CC3311"
INK = "#222222"
GRID = "#DDDDDD"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def _recessive(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _load():
    return {k: pd.read_csv(DATA_DIR / f"de_{k}.csv").set_index("Protein")
            for k in CONTRASTS}


def plot_volcano(results):
    fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 5.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, r) in zip(axes, results.items()):
        _recessive(ax)
        a, b, desc = CONTRASTS[key]
        y = -np.log10(r["t_p"])
        ns = ~r["changed"]
        up = r["changed"] & (r["log2FC"] > 0)
        dn = r["changed"] & (r["log2FC"] < 0)
        ax.scatter(r.loc[ns, "log2FC"], y[ns], s=10, color=C_NS, alpha=0.5,
                   linewidth=0, zorder=2)
        ax.scatter(r.loc[up, "log2FC"], y[up], s=16, color=C_UP, alpha=0.85,
                   linewidth=0, zorder=3, label=f"up ({int(up.sum())})")
        ax.scatter(r.loc[dn, "log2FC"], y[dn], s=16, color=C_DOWN, alpha=0.85,
                   linewidth=0, zorder=3, label=f"down ({int(dn.sum())})")

        # FDR<0.05 boundary drawn as the -log10(p) of the largest changed p.
        if r["changed"].any():
            p_cut = r.loc[r["changed"], "t_p"].max()
            ax.axhline(-np.log10(p_cut), color=C_THRESH, linestyle="--",
                       linewidth=1.2, zorder=1)
            ax.text(ax.get_xlim()[1], -np.log10(p_cut),
                    f"  FDR={FDR_CHANGED:g}", color=C_THRESH, va="bottom",
                    ha="right", fontsize=8)
        ax.axvline(0, color="#888888", linewidth=0.6, zorder=1)

        for g in MARKERS[key]:
            if g in r.index:
                ax.annotate(g, (r.at[g, "log2FC"], -np.log10(r.at[g, "t_p"])),
                            fontsize=8, fontweight="bold",
                            xytext=(3, 3), textcoords="offset points", zorder=4)
        ax.set_title(f"{desc}\n{a} vs {b}  ·  {int(r['changed'].sum())} changed")
        ax.set_xlabel("log2 fold-change (NPX)")
        ax.set_ylabel("-log10(p)")
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("Differential protein abundance by contrast "
                 "(labelled genes = abstract markers)",
                 y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "de_volcano.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overlap(results):
    pre = set(results["pre_heat"].index[results["pre_heat"]["changed"]])
    post = set(results["post_heat"].index[results["post_heat"]["changed"]])
    pre_only, both, post_only = len(pre - post), len(pre & post), len(post - pre)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.add_patch(Circle((4, 3.5), 2.4, facecolor=C_DOWN, alpha=0.25,
                         edgecolor=C_DOWN, linewidth=1.5))
    ax.add_patch(Circle((6, 3.5), 2.4, facecolor=C_UP, alpha=0.25,
                         edgecolor=C_UP, linewidth=1.5))
    ax.text(2.9, 3.5, pre_only, ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(5.0, 3.5, both, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(7.1, 3.5, post_only, ha="center", va="center", fontsize=20, fontweight="bold",
            color=C_UP)
    ax.text(2.9, 6.0, "Pre-acclimation\nheat response\n(PT1 vs PR1)",
            ha="center", va="center", fontsize=10, color=C_DOWN, fontweight="bold")
    ax.text(7.1, 6.0, "Post-acclimation\nheat response\n(PT2 vs PR2)",
            ha="center", va="center", fontsize=10, color=C_UP, fontweight="bold")
    ax.text(7.1, 1.0, "unique to\npost-acclimation", ha="center", va="center",
            fontsize=9, color=C_UP, style="italic")
    ax.set_title(f"Overlap of heat-stress responses  "
                 f"(changed = BH-FDR < {FDR_CHANGED:g})\n"
                 f"{post_only} proteins changed only after acclimation  "
                 f"[abstract: 180]",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "de_overlap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary(results):
    summary = pd.read_csv(DATA_DIR / "de_summary.csv")
    post = set(results["post_heat"].index[results["post_heat"]["changed"]])
    pre = set(results["pre_heat"].index[results["pre_heat"]["changed"]])
    unique_post = len(post - pre)

    col_labels = ["Contrast", "Recreated (BH-FDR < 0.05)", "Abstract"]
    col_widths = [0.24, 0.36, 0.40]
    wrap = [30, 44, 48]
    rows = []
    for _, s in summary.iterrows():
        recreated = (f"{s.changed_fdr05} changed ({s.up} up / {s.down} down); "
                     f"{s.strict_fdr01} at FDR<{FDR_STRICT:g}")
        if s.contrast == "post_heat":
            recreated += f"; {unique_post} unique to post"
        rows.append([f"{s.comparison}\n{s.description}", recreated, s.abstract])

    cell_text = [[textwrap.fill(c, w) for c, w in zip(row, wrap)] for row in rows]
    row_lines = [1] + [max(c.count("\n") + 1 for c in r) for r in cell_text]
    unit = 0.9 / sum(row_lines)

    fig, ax = plt.subplots(figsize=(14, 0.42 * sum(row_lines) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, colWidths=col_widths,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#FFFFFF"); cell.set_linewidth(1.2)
        cell.set_height(row_lines[rr] * unit)
        cell.set_text_props(va="center")
        if rr == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", fontweight="bold", va="center")
        else:
            cell.set_facecolor("#F5F5F5" if rr % 2 else "#FFFFFF")
            if cc == 0:
                cell.set_text_props(fontweight="bold", va="center")
    ax.set_title("Recreation vs. abstract — paired t-test, BH-FDR\n"
                 "(counts differ because the original's statistical method is "
                 "unstated; markers and direction reproduce)",
                 fontsize=12, fontweight="bold", pad=16)
    fig.tight_layout()
    out = RESULTS_DIR / "de_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    results = _load()
    outs = [plot_volcano(results), plot_overlap(results), plot_summary(results)]
    print("Saved figures:")
    for o in outs:
        print(" -", o)


if __name__ == "__main__":
    main()
