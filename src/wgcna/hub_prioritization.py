"""Prioritize hub proteins for physiological regulation of sweating (impute).

Hubs are drawn ONLY from modules passing all three proposal criteria -- the same
screen as enrichment.py, read from its `module_selection.csv` so the two cannot
drift apart. Criterion 3 (MM-GS) matters especially here: in a module with low
MM-GS, module membership and sweat association are unrelated, so a high-|MM|
high-|GS| protein there is not the evidence it appears to be. `PT1/PT2 darkgrey`
(MM-GS 0.162) is excluded for exactly this reason.

Within each selected module, a hub is a protein that is BOTH a core module member
(high |Module Membership| = |kME|) AND strongly sweat-correlated (high |Gene
Significance|). Hubs are ranked by |MM| * |GS| and consolidated across the
sweat modules (kept once, at their strongest module).

Outputs (results/wgcna/impute/):
  hub_prioritization.csv   -- ranked hub table, with each hub's leading
                              GO/KEGG term from its own module's enrichment
  hub_prioritization.png   -- top hubs (|GS| bars coloured by module, |MM| noted)

Run with the Python 3.9 interpreter (PyWGCNA).
"""

import re
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
from wgcna_evaluation import module_membership, gene_significance
import model_development as md

CASE = "impute"
CASE_DIR = DATA_DIR / "wgcna" / CASE
RESULTS_DIR = PROJECT_DIR / "results" / "wgcna" / CASE
SELECTION = RESULTS_DIR / "enrichment" / "module_selection.csv"
TRAIT = "LSR (mg/min/cm2)"
CODES = {"full": None, "PR1/PT1": ["PR1", "PT1"], "PR2/PT2": ["PR2", "PT2"],
         "PT1/PT2": ["PT1", "PT2"], "PR1/PR2": ["PR1", "PR2"]}
MM_HUB, GS_HUB, TOP_N = 0.70, 0.50, 20     # hub thresholds + how many to plot



def _annotate(hubs):
    """Attach each hub's leading GO/KEGG term, taken from its own module's
    enrichment result (module-level tests have the power that a 20-gene test
    does not). Blank where no significant term contains the protein."""
    cache, terms, fdrs = {}, [], []
    for _, r in hubs.iterrows():
        key = (r["contrast"], r["module"])
        if key not in cache:
            f = (RESULTS_DIR / "enrichment" /
                 f"{r['contrast'].replace('/', '_')}_{r['module']}_enrichment.csv")
            cache[key] = pd.read_csv(f) if f.exists() else None
        d = cache[key]
        hit = None
        if d is not None:
            m = d[d["Genes"].astype(str).str.split(";").apply(
                lambda g: r["protein"] in g)]
            if len(m):
                hit = m.loc[m["Adjusted P-value"].idxmin()]
        terms.append(re.sub(r"\s*\(GO:\d+\)$", "", hit["Term"]) if hit is not None else "")
        fdrs.append(f"{hit['Adjusted P-value']:.1e}" if hit is not None else "")
    return hubs.assign(top_term=terms, top_term_fdr=fdrs)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    if not SELECTION.exists():
        sys.exit(f"missing {SELECTION}\nrun:  python enrichment.py {CASE}")
    sel = pd.read_csv(SELECTION)
    sel = sel[sel["selected"]]
    print(f"modules passing all three criteria: "
          f"{', '.join(sel.network + ' ' + sel.module)}\n")

    rows = []
    for name, grp in sel.groupby("network", sort=False):
        codes = CODES[name]
        mask = None if codes is None else pd.Series(ex.isin(codes), index=tr.index)
        obj = md.build(expr, tr, mask, name.replace("/", "_"))
        for mod in grp["module"]:
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
    hubs = _annotate(hubs)
    hubs.to_csv(RESULTS_DIR / "hub_prioritization.csv", index=False)

    print(f"prioritized {len(hubs)} hub proteins "
          f"(|MM|>={MM_HUB} & |GS|>={GS_HUB}) across the modules passing all criteria\n")
    print(hubs.head(TOP_N).to_string(index=False))
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
    labels = [f"{g}  (MM={mm:.2f})" for g, mm in zip(top["protein"], top["MM"])]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|Gene Significance| (correlation with LSR)")
    ax.set_title("Prioritized hub proteins for sweating (LSR)\n"
                 "|MM| ≥ 0.70 & |GS| ≥ 0.50, ranked by |MM|·|GS|",
                 fontsize=10, fontweight="bold")
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
