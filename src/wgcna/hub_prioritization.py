"""Prioritize hub proteins for physiological regulation of sweating (impute).

Within each module whose eigengene correlates with local sweat rate (LSR) at
BH-FDR < 0.05, a hub is a protein that is BOTH a core module member (high
|Module Membership| = |kME|) AND strongly sweat-correlated (high |Gene
Significance|). Hubs are ranked by |MM| * |GS| and consolidated across the
sweat modules (kept once, at their strongest module). Proteins with an
established role in sweating / thermoregulation are flagged.

Outputs (results/wgcna/impute/):
  hub_prioritization.csv   -- ranked hub table
  hub_prioritization.png   -- top hubs (|GS| bars coloured by module, |MM| noted)

Run with the Python 3.9 interpreter (PyWGCNA).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from wgcna_evaluation import module_trait_table, module_membership, gene_significance
import model_development as md

CASE_DIR = DATA_DIR / "wgcna" / "impute"
RESULTS_DIR = PROJECT_DIR / "results" / "wgcna" / "impute"
TRAIT = "LSR (mg/min/cm2)"
CONTRASTS = {"PR2/PT2": ["PR2", "PT2"], "PT1/PT2": ["PT1", "PT2"]}
MT_CUT, FDR_MODULE = 0.5, 0.05
MM_HUB, GS_HUB, TOP_N = 0.70, 0.50, 20     # hub thresholds + how many to plot

# Compact, established set of sweat / thermoregulation-relevant genes (water &
# ion transport, sweat-gland secretory, autonomic, skin blood-flow / vascular).
SWEAT_GENES = {
    "AQP1", "AQP3", "AQP5", "CA2", "CA6", "CA12",
    "KLK1", "KLK7", "KLK11", "KLK13", "KLK14",
    "ATP1A1", "ATP1B1", "SLC12A2", "CFTR", "SCNN1A", "SCNN1B", "SCNN1G",
    "ANO1", "BEST2", "DCD", "PIP", "SCGB2A2", "MUC7",
    "CHRM3", "ADRB2", "EDN1", "NOS3", "ACE", "ACE2", "VEGFA", "TIMP1",
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    rows = []
    for name, codes in CONTRASTS.items():
        obj = md.build(expr, tr, pd.Series(ex.isin(codes), index=tr.index),
                       name.replace("/", "_"))
        mt = module_trait_table(obj, TRAIT)
        sig = mt[(mt["r"].abs() >= MT_CUT) & (mt["FDR"] < FDR_MODULE)]
        for mod in sig.index:
            MM = module_membership(obj, mod)
            GS = gene_significance(obj, TRAIT, MM.index)
            for g in MM.index:
                if abs(MM[g]) >= MM_HUB and abs(GS[g]) >= GS_HUB:
                    rows.append({"protein": g, "contrast": name, "module": mod,
                                 "MM": round(float(MM[g]), 3), "GS": round(float(GS[g]), 3),
                                 "score": round(abs(MM[g]) * abs(GS[g]), 3)})

    hubs = pd.DataFrame(rows).sort_values("score", ascending=False)
    # keep each protein once, at its strongest module
    hubs = hubs.drop_duplicates("protein", keep="first").reset_index(drop=True)
    hubs["known_sweat_gene"] = hubs["protein"].isin(SWEAT_GENES)
    hubs.to_csv(RESULTS_DIR / "hub_prioritization.csv", index=False)

    print(f"prioritized {len(hubs)} hub proteins "
          f"(|MM|>={MM_HUB} & |GS|>={GS_HUB}) across the significant LSR modules\n")
    print(hubs.head(TOP_N).to_string(index=False))
    known = hubs[hubs["known_sweat_gene"]]
    print(f"\nwith an established sweat/thermoregulation role: "
          f"{list(known['protein']) or 'none'}")

    _plot(hubs.head(TOP_N))


def _plot(top):
    top = top.iloc[::-1]                       # highest score at the top
    mods = sorted(top["module"].unique())
    palette = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(mods)}
    colors = [cmap[m] for m in top["module"]]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 1.5))
    y = range(len(top))
    ax.hlines(y, 0, top["GS"].abs(), color=colors, linewidth=2, zorder=1)
    ax.scatter(top["GS"].abs(), y, color=colors, s=60, zorder=2,
               edgecolor="white", linewidth=0.6)
    labels = [f"{'* ' if k else ''}{g}  (MM={mm:.2f})"
              for g, mm, k in zip(top["protein"], top["MM"], top["known_sweat_gene"])]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|Gene Significance| (correlation with LSR)")
    ax.set_title("Prioritized hub proteins for sweating (LSR)\n"
                 "|MM| ≥ 0.70 & |GS| ≥ 0.50, ranked by |MM|·|GS|;  "
                 "* = known sweat/thermoregulation protein", fontsize=10, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=cmap[m], label=m)
               for m in mods]
    ax.legend(handles=handles, title="module", fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = RESULTS_DIR / "hub_prioritization.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("\nsaved:", RESULTS_DIR / "hub_prioritization.csv")
    print("saved:", out)


if __name__ == "__main__":
    main()
