"""GO + KEGG enrichment of the significant sweat (LSR) modules.

Scope: the chosen **impute 10%** workflow, with **local sweat rate (LSR)** as the
sweat outcome. Whole-body 'Sweat rate (L/h)' is not screened, and the complete-case
workflow is a sensitivity check only (`python enrichment.py complete`), not the
headline result.

A module is SELECTED only when it passes **all three proposal criteria**:

  1. topology     -- scale-free fit R^2 > 0.8 at the selected soft-power;
  2. module-trait -- |r| >= 0.5 and BH-FDR < 0.05 vs LSR;
  3. MM vs GS     -- |r| > 0.6 and p < 0.05 within the module.

This selects **three** modules, in the two heat contrasts:

  PT1/PT2 dimgrey (MM-GS 0.837) · PT1/PT2 lightgrey (0.666) · PR2/PT2 black (0.762)

Two near-misses, both excluded and both worth knowing:

  * `PT1/PT2 darkgrey`  -- clears criteria 1-2 (r = +0.528, FDR = 0.028) but its
    MM-GS is only **0.162**: within that module, how central a protein is says
    essentially nothing about how sweat-associated it is.
  * `full black`        -- clears FDR (0.004) but r = 0.485 < 0.5. The full network
    also pools all 40 samples across all four exposures, so its LSR correlation is
    confounded by thermal stage; only the contrast networks are interpretable.

Criterion 3 uses the proposal's original **0.6**, not the relaxed 0.5 -- all three
selected modules have MM-GS >= 0.666, so the two cut-offs give an identical result
and there is no deviation from the proposal to defend. See mmgs_threshold.py.

Every module keeps its per-criterion flags (`passes_topology`, `passes_mt_r`,
`passes_fdr`, `passes_mmgs`) in `module_selection.csv`, so the excluded ones stay
auditable.

For each selected module, run Enrichr (GO Biological Process + KEGG Human) on the
module's proteins, save the significant terms, and flag terms matching the target
themes (seven topics chosen in advance; see THEMES).

Usage (Python 3.9 with PyWGCNA + gseapy + internet):
  python enrichment.py impute      # -> results/wgcna/impute/enrichment/   (primary)
  python enrichment.py complete    # -> results/wgcna/complete/enrichment/ (sensitivity)
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from wgcna_evaluation import (module_trait_table, module_membership,
                              gene_significance, scale_free_fit)
import model_development as md

LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
TRAIT = "LSR (mg/min/cm2)"     # primary sweat outcome; whole-body is not screened
NETWORKS = {"full": None,
            "PR1/PT1": ["PR1", "PT1"], "PR2/PT2": ["PR2", "PT2"],
            "PT1/PT2": ["PT1", "PT2"], "PR1/PR2": ["PR1", "PR2"]}
R2_CUT = 0.8                     # criterion 1: scale-free topology fit
MT_CUT, FDR_MODULE = 0.5, 0.05   # criterion 2: |module-trait r| & BH-FDR
MMGS_CUT, MMGS_P = 0.6, 0.05     # criterion 3: MM vs GS (proposal value)
THEMES = {
    "heat adaptation": ["heat", "temperature", "thermogen", "cold", "brown fat",
                        "thermoregulat"],
    "stress response": ["stress", "unfolded protein", "heat shock", "chaperone",
                        "oxidative", "hypoxia", "hif", "reactive oxygen"],
    "immune": ["immune", "inflammat", "cytokine", "interleukin", "leukocyte",
               "complement", "lymphocyte", "innate", "interferon", "chemokine",
               "t cell", "b cell"],
    "blood vessels": ["vascular", "angiogen", "blood vessel", "endothel",
                      "vasoconstric", "vasodilat", "shear", "atheroscler", "vegf",
                      "blood pressure"],
    "salt transport": ["ion transport", "sodium", "potassium", "chloride",
                       "electrolyte", "cation", "anion", "ion homeostasis"],
    # named metabolic pathways only. The bare words "metabolic"/"metabolism" are
    # avoided on purpose: in GO they mostly appear in regulatory boilerplate such
    # as "Regulation Of Macromolecule Metabolic Process", which is not metabolism
    # in the physiological sense being asked about here.
    "metabolism": ["glycolysis", "gluconeogen", "lipid", "fatty acid", "cholesterol",
                   "insulin", "glucose", "oxidative phosphoryl", "citrate cycle",
                   "tca cycle", "lipoprotein", "adipo", "glycogen", "steroid",
                   "purine", "pyrimidine", "amino acid metabolic", "mitochondri",
                   "atp"],
    "neurological signaling": ["neuro", "synap", "axon", "nerve", "glial", "myelin",
                               "dendrit", "neurotransmitter", "acetylcholine",
                               "adrenergic", "catecholamine"],
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


def screen_modules(case, expr, tr, ex):
    """Score every module in every network against the sweat trait (LSR).

    A module is selected only if it passes ALL THREE proposal criteria (topology,
    module-trait, MM-GS). Per-criterion flags are kept for every module so the
    near-misses stay auditable.

    Returns (scan, targets): `scan` is the full audit table (one row per network
    x module); `targets` is the list of (label, genes) to enrich.
    """
    rows, genes_of = [], {}
    for net, codes in NETWORKS.items():
        mask = None if codes is None else pd.Series(ex.isin(codes), index=tr.index)
        obj = md.build(expr, tr, mask, net.replace("/", "_"))
        colors = obj.datExpr.var["moduleColors"]
        power, r2 = scale_free_fit(obj)
        mt = module_trait_table(obj, TRAIT)
        for mod in mt.index:
            MM = module_membership(obj, mod)
            GS = gene_significance(obj, TRAIT, MM.index)
            mmgs_r, mmgs_p = pearsonr(MM.values, GS.values)
            genes_of[(net, mod)] = list(colors.index[colors == mod])
            rows.append({
                "case": case, "network": net, "module": mod,
                "n_prot": int((colors == mod).sum()),
                "power": power, "r2": round(float(r2), 3),
                "mt_r": round(float(mt.loc[mod, "r"]), 3),
                "mt_FDR": round(float(mt.loc[mod, "FDR"]), 4),
                "mmgs_r": round(float(mmgs_r), 3), "mmgs_p": mmgs_p,
                "passes_topology": bool(r2 > R2_CUT),
                "passes_fdr": bool(mt.loc[mod, "FDR"] < FDR_MODULE),
                "passes_mt_r": bool(abs(mt.loc[mod, "r"]) >= MT_CUT),
                "passes_mmgs": bool(abs(mmgs_r) > MMGS_CUT and mmgs_p < MMGS_P),
            })

    scan = pd.DataFrame(rows)
    scan["selected"] = (scan["passes_topology"] & scan["passes_fdr"]
                        & scan["passes_mt_r"] & scan["passes_mmgs"])
    scan = scan.sort_values(["selected", "mt_FDR"], ascending=[False, True])

    targets = []
    for _, r in scan[scan["selected"]].iterrows():
        targets.append((f"{r.network} {r.module}", genes_of[(r.network, r.module)]))
        print(f"[{case}] {r.network:8s} '{r.module}': {r.n_prot} proteins "
              f"(R²={r.r2:.2f}, r={r.mt_r:+.2f}, FDR={r.mt_FDR:.3f}, "
              f"MM-GS={r.mmgs_r:+.2f})")

    near = scan[~scan["selected"] & scan["passes_fdr"] & scan["passes_mt_r"]]
    for _, r in near.iterrows():
        why = []
        if not r.passes_topology:
            why.append(f"R²={r.r2:.3f} fails >{R2_CUT}")
        if not r.passes_mmgs:
            why.append(f"MM-GS={r.mmgs_r:+.3f} fails |r|>{MMGS_CUT}")
        print(f"[{case}] EXCLUDED {r.network} '{r.module}': clears criterion 2 "
              f"(r={r.mt_r:+.2f}, FDR={r.mt_FDR:.3f}) but {' and '.join(why)}")
    return scan, targets


def run_case(case):
    out = _out_dir(case)
    out.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(_case_dir(case) / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(_case_dir(case) / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    scan, targets = screen_modules(case, expr, tr, ex)
    scan.to_csv(out / "module_selection.csv", index=False)
    print(f"\n[{case}] screened {len(scan)} module x trait pairs -> "
          f"{len(targets)} gene sets to enrich "
          f"(saved {out / 'module_selection.csv'})\n")

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
    if n == 0:                      # no module passed all criteria (complete case)
        print(f"[{case}] no modules passed all criteria — no figure written")
        return
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
        sel_f = _out_dir(case) / "module_selection.csv"
        sel = pd.read_csv(sel_f) if sel_f.exists() else pd.DataFrame()
        for csv in sorted(_out_dir(case).glob("*_enrichment.csv")):
            df = pd.read_csv(csv)
            # re-derive from Term: THEMES may have changed since the file was written
            df["theme"] = df["Term"].map(themes_of)
            df.to_csv(csv, index=False)
            mod = csv.stem.replace("_enrichment", "")
            # module_selection keys on (network, module); the filename joins them
            passes = True
            if len(sel):
                key = sel[(sel.network.str.replace("/", "_") + "_" + sel.module) == mod]
                passes = bool(key["passes_mmgs"].all()) if len(key) else True
            counts = {th: int(df["theme"].fillna("").str.contains(th).sum()) for th in THEMES}
            rows.append({"case": case, "module": mod, "passes_mmgs": passes,
                         "n_terms": len(df), **counts})
    tab = pd.DataFrame(rows)
    out = PROJECT_DIR / "results" / "wgcna" / "enrichment_theme_summary.csv"
    tab.to_csv(out, index=False)

    # heatmap figure: rows = module, cols = themes, cell = # matching terms
    label = tab["case"] + " · " + tab["module"] + tab["passes_mmgs"].map(
        {True: "", False: "  (fails MM-GS)"})
    mat = tab.set_index(label)[list(THEMES)]
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
