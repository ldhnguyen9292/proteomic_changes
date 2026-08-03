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
  hub_enrichment.png                      3-panel figure (rates, themes, terms)

Usage (Python 3.8+ with gseapy + internet):
  python hub_enrichment.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


C_HUB, C_FULL, C_THEMED, C_PLAIN = "#0072B2", "#BBBBBB", "#D55E00", "#9C9C9C"


def _plot(rows, theme_rows, per_module):
    """A: hub vs full match rate. B: theme counts. C: top terms per module."""
    short = {k: k.split(". ", 1)[1] for k in THEMES}
    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.5],
                          hspace=0.42, wspace=0.26, top=0.86, bottom=0.07,
                          left=0.06, right=0.975)

    # ---- A. match rate, same vocabulary both sides -------------------------
    axA = fig.add_subplot(gs[0, 0])
    mods = [r["module"] for r in rows]
    x = np.arange(len(mods)); w = 0.36
    hub = [r["hub_rate"] for r in rows]
    full = [r["full_rate"] for r in rows]
    axA.bar(x - w/2, hub, w, color=C_HUB, label="hub proteins only", zorder=3)
    axA.bar(x + w/2, full, w, color=C_FULL, label="whole module", zorder=3)
    for i, (h, f, r) in enumerate(zip(hub, full, rows)):
        axA.text(i - w/2, h + 1.4, f"{h:.0f}%", ha="center", fontsize=10,
                 fontweight="bold", color=C_HUB)
        axA.text(i + w/2, f + 1.4, f"{f:.0f}%", ha="center", fontsize=10,
                 color="#666")
        arrow = "▲" if h > f else "▼"
        axA.text(i, max(h, f) + 7, arrow, ha="center", fontsize=13,
                 color="#2E7D32" if h > f else "#C0392B")
    axA.set_xticks(x)
    axA.set_xticklabels([f"{r['module']}\n{r['hubs']} hubs / {r['hub_terms']} terms"
                         for r in rows], fontsize=9.5)
    axA.set_ylabel("terms matching ≥ 1 of the 7 themes (%)", fontsize=9.5)
    axA.set_ylim(0, 78)
    axA.legend(fontsize=9, frameon=False, loc="upper right")
    axA.set_title("A. Does restricting to hubs improve the match rate?\n"
                  "both sides scored with the SAME extended vocabulary",
                  fontsize=10.5, fontweight="bold", loc="left")
    axA.grid(axis="y", color="#EEEEEE", zorder=0); axA.set_axisbelow(True)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)

    # ---- B. theme counts, hubs only ----------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    M = np.array([[d[th] for th in THEMES] for d in theme_rows], float)
    im = axB.imshow(M, cmap="Oranges", aspect="auto", vmin=0, vmax=max(M.max(), 1))
    axB.set_xticks(range(len(THEMES)))
    axB.set_xticklabels([short[t] for t in THEMES], rotation=28, ha="right",
                        fontsize=8.5)
    axB.set_yticks(range(len(theme_rows)))
    axB.set_yticklabels([f"{d['module']}  ({d['hub_terms']} terms)"
                         for d in theme_rows], fontsize=9.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            axB.text(j, i, v, ha="center", va="center", fontsize=10,
                     fontweight="bold" if v else "normal",
                     color="white" if v > M.max() * 0.6 else ("#CCC" if not v else "#222"))
    for j, th in enumerate(THEMES):
        if M[:, j].sum() == 0:
            axB.add_patch(plt.Rectangle((j - .5, -.5), 1, M.shape[0],
                                        fill=False, edgecolor="#C0392B", lw=2.2))
    axB.set_title("B. Theme counts for the hub proteins\n"
                  "red outline = theme with no hits in any module",
                  fontsize=10.5, fontweight="bold", loc="left")
    fig.colorbar(im, ax=axB, shrink=0.72, label="terms")

    # ---- C. strongest terms per module -------------------------------------
    gsC = gs[1, :].subgridspec(1, 3, wspace=0.62)
    for k, (m, (hub_res, _)) in enumerate(per_module.items()):
        ax = fig.add_subplot(gsC[0, k])
        top = hub_res.head(7).iloc[::-1]
        if len(top):
            v = -np.log10(top["Adjusted P-value"].values)
            cols = [C_THEMED if t else C_PLAIN for t in top["theme"].fillna("")]
            ax.barh(range(len(top)), v, color=cols, zorder=3)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([t[:44] for t in top["Term"]], fontsize=7.8)
            for i, val in enumerate(v):
                ax.text(val + 0.06, i, f"{val:.1f}", va="center", fontsize=7.5,
                        color="#555")
        ax.axvline(-np.log10(0.05), color="#C0392B", ls="--", lw=1.2)
        ax.text(-np.log10(0.05), len(top) - 0.4, " FDR 0.05", fontsize=7.5,
                color="#C0392B", va="top")
        ax.set_xlabel("−log10(FDR)", fontsize=8.5)
        ax.set_xlim(0, max(4.2, (-np.log10(hub_res["Adjusted P-value"].min())) * 1.25))
        ax.set_title(f"{m} — {len(hub_res)} significant terms", fontsize=10,
                     fontweight="bold")
        ax.grid(axis="x", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.text(0.06, 0.475, "C. Strongest hub terms — orange = maps to a theme, "
             "grey = maps to none", fontsize=10.5, fontweight="bold")
    fig.suptitle("Hub-protein enrichment: does focusing on hubs sharpen the "
                 "mapping to the seven themes?", fontsize=14, fontweight="bold",
                 y=0.955)
    f = OUT / "hub_enrichment.png"
    fig.savefig(f, bbox_inches="tight", dpi=165)
    plt.close(fig)
    print("saved:", f)


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

    _plot(rows, theme_rows, per_module)

    print(f"\n{'='*70}\n{summary.to_string(index=False)}")
    print(f"\n{themes_tab.to_string(index=False)}")
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
