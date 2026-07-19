"""GO + KEGG enrichment of the significant sweat (LSR) modules.

For each module whose eigengene correlates with local sweat rate (LSR) at
BH-FDR < 0.05, run Enrichr (GO Biological Process + KEGG Human) on the module's
proteins, save the significant terms, and flag terms matching the target themes
(heat adaptation, stress response, immune, vascular, electrolyte transport).
For the impute workflow the full-network 'black' heat-response module is also
included for context.

Usage (Python 3.9 with PyWGCNA + gseapy + internet):
  python enrichment.py impute      # -> results/wgcna/impute/enrichment/
  python enrichment.py complete    # -> results/wgcna/complete/enrichment/
  python enrichment.py summary     # module x theme table across both cases

Note: uses Enrichr's default background gene set.
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
import gseapy as gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from wgcna_evaluation import module_trait_table
import model_development as md

LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
TRAIT = "LSR (mg/min/cm2)"
CONTRASTS = {"PR2/PT2": ["PR2", "PT2"], "PT1/PT2": ["PT1", "PT2"]}
MT_CUT, FDR_MODULE = 0.5, 0.05
THEMES = {
    "heat adaptation": ["heat", "temperature", "thermogen", "cold", "brown fat"],
    "stress response": ["stress", "unfolded protein", "heat shock", "chaperone",
                         "oxidative", "hypoxia", "hif", "reactive oxygen"],
    "immune": ["immune", "inflammat", "cytokine", "interleukin", "leukocyte",
               "complement", "lymphocyte", "innate", "interferon", "chemokine",
               "t cell", "b cell"],
    "vascular": ["vascular", "angiogen", "blood vessel", "endothel", "vasoconstric",
                 "vasodilat", "shear", "atheroscler", "vegf", "blood pressure"],
    "electrolyte transport": ["ion transport", "sodium", "potassium", "chloride",
                              "electrolyte", "cation", "anion", "ion homeostasis"],
}


def _case_dir(case):
    return DATA_DIR / "wgcna" / case


def _out_dir(case):
    return PROJECT_DIR / "results" / "wgcna" / case / "enrichment"


def themes_of(term):
    t = term.lower()
    return ";".join(k for k, kws in THEMES.items() if any(kw in t for kw in kws))


def enrich(genes):
    res = gp.enrichr(gene_list=list(genes), gene_sets=LIBRARIES,
                     organism="human", outdir=None).results
    res = res[res["Adjusted P-value"] < 0.05].copy()
    res["theme"] = res["Term"].map(themes_of)
    return res.sort_values(["Gene_set", "Adjusted P-value"])


def run_case(case):
    out = _out_dir(case)
    out.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(_case_dir(case) / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(_case_dir(case) / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    targets = []   # (label, genes)
    for name, codes in CONTRASTS.items():
        obj = md.build(expr, tr, pd.Series(ex.isin(codes), index=tr.index),
                       name.replace("/", "_"))
        mt = module_trait_table(obj, TRAIT)
        sig = mt[(mt["r"].abs() >= MT_CUT) & (mt["FDR"] < FDR_MODULE)]
        for mod in sig.index:
            genes = list(obj.datExpr.var.index[obj.datExpr.var["moduleColors"] == mod])
            targets.append((f"{name} {mod}", genes))
            print(f"[{case}] sig LSR module {name} '{mod}': {len(genes)} proteins "
                  f"(r={mt.loc[mod, 'r']:+.2f}, FDR={mt.loc[mod, 'FDR']:.3f})")

    if case == "impute":                     # add the full-network heat-response module
        full = md.build(expr, tr, None, "full")
        if "black" in set(full.datExpr.var["moduleColors"]):
            genes = list(full.datExpr.var.index[full.datExpr.var["moduleColors"] == "black"])
            targets.append(("full black", genes))
            print(f"[{case}] heat-response module full 'black': {len(genes)} proteins")

    results = {}
    for label, genes in targets:
        res = enrich(genes)
        res.to_csv(out / f"{label.replace('/', '_').replace(' ', '_')}_enrichment.csv",
                   index=False)
        results[label] = res
        print(f"\n=== [{case}] {label} ({len(genes)} proteins) — {len(res)} sig terms ===")
        for lib in LIBRARIES:
            for _, row in res[res.Gene_set == lib].head(5).iterrows():
                tag = f"  <{row['theme']}>" if row["theme"] else ""
                print(f"  {lib.split('_20')[0][:4]} | {row['Term'][:52]:52s} "
                      f"FDR={row['Adjusted P-value']:.1e}{tag}")
    _plot(case, results)


def _plot(case, results):
    labels = list(results)
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.0 * n), squeeze=False)
    for i, label in enumerate(labels):
        res = results[label]
        for j, lib in enumerate(LIBRARIES):
            ax = axes[i][j]
            top = res[res.Gene_set == lib].head(8).iloc[::-1]
            if len(top):
                ax.barh(range(len(top)), -np.log10(top["Adjusted P-value"].values),
                        color=["#D55E00" if th else "#0072B2" for th in top["theme"]],
                        alpha=0.85)
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels([t[:50] for t in top["Term"]], fontsize=7)
            else:
                ax.text(0.5, 0.5, "no significant terms", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#888")
            ax.set_xlabel("-log10(FDR)", fontsize=8)
            ax.set_title(f"{label}  —  {lib.split('_20')[0].replace('_', ' ')}",
                         fontsize=9, fontweight="bold")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle(f"GO / KEGG enrichment — {case} (orange = target theme)",
                 fontsize=13, fontweight="bold", y=1.004)
    fig.tight_layout()
    f = _out_dir(case) / "enrichment_top_terms.png"
    fig.savefig(f, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("\nsaved figure:", f)


def theme_summary():
    """Read every enrichment CSV across cases -> module x theme count table."""
    rows = []
    for case in ("impute", "complete"):
        for csv in sorted(_out_dir(case).glob("*_enrichment.csv")):
            df = pd.read_csv(csv)
            mod = csv.stem.replace("_enrichment", "")
            counts = {th: int(df["theme"].fillna("").str.contains(th).sum()) for th in THEMES}
            rows.append({"case": case, "module": mod, "n_terms": len(df), **counts})
    tab = pd.DataFrame(rows)
    out = PROJECT_DIR / "results" / "wgcna" / "enrichment_theme_summary.csv"
    tab.to_csv(out, index=False)

    # heatmap figure: rows = module, cols = themes, cell = # matching terms
    mat = tab.set_index(tab["case"] + " · " + tab["module"])[list(THEMES)]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(mat) + 2))
    im = ax.imshow(mat.values, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(THEMES)))
    ax.set_xticklabels(list(THEMES), rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(len(mat)):
        for j in range(len(THEMES)):
            v = mat.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=8,
                    color="white" if v > mat.values.max() * 0.5 else "#222")
    ax.set_title("Enriched terms per theme (module × theme)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="# significant terms")
    fig.tight_layout()
    f = PROJECT_DIR / "results" / "wgcna" / "enrichment_theme_summary.png"
    fig.savefig(f, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(tab.to_string(index=False))
    print("\nsaved:", out, "\nsaved:", f)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "impute"
    if arg == "summary":
        theme_summary()
    else:
        run_case(arg)
