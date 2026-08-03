"""Enrichment of the HUB proteins only, with extended-synonym theme mapping.

Rationale: the full-module enrichment matches the seven target themes for only
26-40% of its terms, partly because module lists are large and partly because
strict keyword matching misses synonyms. This script tests whether restricting
to hub proteins, and widening the vocabulary, gives a cleaner mapping.

IMPORTANT CONTROL. Widening the vocabulary raises the match rate on its own,
whatever gene list is used. So the script re-scores the FULL-module enrichment
with exactly the same extended themes. Only the hub-vs-full difference measured
under the same vocabulary says anything about hub focus; comparing hub+extended
against full+strict would be measuring two changes at once.

Statistical caveat: enrichment power scales with list size. M3 has 10 hubs and
M1 has 27, so few or no terms may survive FDR < 0.05. That is a property of the
test, not evidence of absent biology, and it is reported rather than hidden.

Outputs (results/wgcna/impute/hub_enrichment/):
  M1_hub_enrichment.csv, M2_..., M3_...   per-module significant terms
  hub_theme_summary.csv                   module x theme counts, hub vs full
  hub_theme_summary.md                    the same table in markdown

Usage (Python 3.8+ with gseapy + internet):
  python hub_enrichment.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import gseapy as gp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import PROJECT_DIR

RES = PROJECT_DIR / "results" / "wgcna" / "impute"
OUT = RES / "hub_enrichment"
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
FDR = 0.05
MODULES = {"M1": "dimgrey", "M2": "black", "M3": "lightgrey"}
FULL_FILE = {"M1": "PT1_PT2_dimgrey", "M2": "PR2_PT2_black", "M3": "PT1_PT2_lightgrey"}

# ---------------------------------------------------------------------------
# Seven target themes with EXTENDED synonyms. Themes are not mutually
# exclusive: a term may map to more than one (e.g. "Fluid shear stress and
# atherosclerosis" -> blood vessels + stress response).
# ---------------------------------------------------------------------------
THEMES = {
    "1. Sweat loss / hydration": [
        "sweat", "perspir", "hydrat", "dehydrat", "water transport", "aquaporin",
        "water homeostasis", "body fluid", "fluid transport", "osmoregulat",
        "exocrine", "salivary", "secretion by cell", "gland development",
    ],
    "2. Heat stress / heat adaptation": [
        "heat", "temperature", "thermal", "thermogen", "thermoregulat",
        "hyperthermi", "acclimat", "cold", "brown fat", "heat shock", "hsp",
        "chaperone", "protein folding", "unfolded protein", "protein refolding",
    ],
    "3. Stress response": [
        "stress", "oxidative", "reactive oxygen", "redox", "antioxidant",
        "glutathione", "hypoxia", "hif", "dna damage", "autophagy", "proteasome",
        "unfolded protein", "cellular response to chemical", "detoxification",
    ],
    "4. Immune response": [
        "immune", "immunit", "inflammat", "cytokine", "chemokine", "interleukin",
        "interferon", "leukocyte", "lymphocyte", "neutrophil", "macrophage",
        "monocyte", "t cell", "b cell", "complement", "innate", "adaptive immun",
        "antigen", "toll-like", "nf-kappab", "tnf", "histocompatibility",
        "phagocyt", "granulocyte", "mast cell", "natural killer",
    ],
    "5. Blood vessels / vascular regulation": [
        "vascular", "vasculature", "angiogen", "blood vessel", "endotheli",
        "vasoconstric", "vasodilat", "nitric oxide", "shear stress", "atheroscler",
        "vegf", "blood pressure", "blood circulation", "hemostasis", "haemostasis",
        "coagulation", "platelet", "capillar", "artery", "arterial",
        "smooth muscle", "heart", "cardiac",
    ],
    "6. Salt / ion transport": [
        "ion transport", "ion transmembrane", "ion homeostasis", "sodium",
        "potassium", "chloride", "calcium transport", "calcium ion",
        "electrolyte", "cation", "anion", "solute carrier", "symport", "antiport",
        "ion channel", "atpase", "membrane potential", "osmotic",
    ],
    "7. Metabolism / neurological signaling": [
        # metabolism
        "glycolysis", "gluconeogen", "lipid", "fatty acid", "cholesterol",
        "insulin", "glucose", "oxidative phosphoryl", "citrate cycle", "tca cycle",
        "lipoprotein", "adipo", "glycogen", "steroid", "purine", "pyrimidine",
        "amino acid metabolic", "mitochondri", "atp metabolic", "energy",
        # neurological signaling
        "neuro", "synap", "axon", "nerve", "glial", "myelin", "dendrit",
        "neurotransmitter", "acetylcholine", "adrenergic", "catecholamine",
        "brain", "neuronal",
    ],
}


def themes_of(term):
    t = str(term).lower()
    return ";".join(k for k, kws in THEMES.items() if any(kw in t for kw in kws))


def enrich(genes):
    """Enrichr on one gene list -> significant terms, theme-annotated."""
    res = gp.enrichr(gene_list=list(genes), gene_sets=LIBRARIES,
                     organism="human", outdir=None).results
    res = res[res["Adjusted P-value"] < FDR].copy()
    res["theme"] = res["Term"].map(themes_of)
    return res.sort_values("Adjusted P-value")


def match_rate(df):
    """(n terms matching >=1 theme, total, percent)."""
    if not len(df):
        return 0, 0, float("nan")
    hit = df["theme"].fillna("").ne("").sum()
    return int(hit), len(df), round(hit / len(df) * 100, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hubs = pd.read_csv(RES / "hub_prioritization.csv")

    rows, per_module = [], {}
    for m, colour in MODULES.items():
        genes = [g.split("__")[0] for g in hubs.loc[hubs.module == colour, "protein"]]
        print(f"\n{'='*70}\n{m}: {len(genes)} hub proteins\n{genes}")

        hub_res = enrich(genes)
        hub_res.to_csv(OUT / f"{m}_hub_enrichment.csv", index=False)
        h_hit, h_tot, h_pct = match_rate(hub_res)

        # same extended vocabulary applied to the full-module result: the control
        full = pd.read_csv(RES / "enrichment" / f"{FULL_FILE[m]}_enrichment.csv")
        full["theme"] = full["Term"].map(themes_of)
        f_hit, f_tot, f_pct = match_rate(full)

        per_module[m] = (hub_res, full)
        rows.append(dict(module=m, hubs=len(genes),
                         hub_terms=h_tot, hub_matched=h_hit, hub_rate=h_pct,
                         full_terms=f_tot, full_matched=f_hit, full_rate=f_pct))
        print(f"  hub terms  (FDR<{FDR}): {h_tot:>4}   matched {h_hit:>4}  ({h_pct}%)")
        print(f"  full terms (same vocab): {f_tot:>4}   matched {f_hit:>4}  ({f_pct}%)")
        for _, r in hub_res.head(5).iterrows():
            print(f"     {r['Adjusted P-value']:.1e}  {r.Term[:56]:<56} {r.theme}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "hub_theme_summary.csv", index=False)

    # per-theme counts, hubs only
    theme_rows = []
    for m, (hub_res, _) in per_module.items():
        d = {"module": m, "hub_terms": len(hub_res)}
        for th in THEMES:
            d[th] = int(hub_res["theme"].fillna("").str.contains(th, regex=False).sum())
        theme_rows.append(d)
    themes_tab = pd.DataFrame(theme_rows)
    themes_tab.to_csv(OUT / "hub_theme_counts.csv", index=False)

    with open(OUT / "hub_theme_summary.md", "w") as fh:
        fh.write("# Hub-protein enrichment vs full-module enrichment\n\n")
        fh.write("Both scored with the **same** extended-synonym themes, so the "
                 "comparison isolates the gene list rather than the vocabulary.\n\n")
        fh.write("| Module | Hubs | Hub terms | Hub match rate | Full terms | "
                 "Full match rate |\n|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['module']} | {r['hubs']} | {r['hub_terms']} | "
                     f"{r['hub_rate']}% | {r['full_terms']} | {r['full_rate']}% |\n")
        fh.write("\n## Theme counts (hub proteins only)\n\n| Module | "
                 + " | ".join(THEMES) + " |\n|" + "---|" * (len(THEMES) + 1) + "\n")
        for d in theme_rows:
            fh.write(f"| {d['module']} | "
                     + " | ".join(str(d[th]) for th in THEMES) + " |\n")

    print(f"\n{'='*70}\n{summary.to_string(index=False)}")
    print(f"\n{themes_tab.to_string(index=False)}")
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
