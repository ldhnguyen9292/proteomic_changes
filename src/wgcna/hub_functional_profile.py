"""Data-driven functional profile of the hub proteins of three named modules.

Self-contained: rebuilds the required WGCNA networks, derives the hub proteins,
fetches official gene names, and queries Enrichr. Contains no keyword list and
no target topics -- the reported terms are whatever comes back at the lowest
FDR, so nothing is pre-selected.

Modules analysed (exact names as given):
    impute · PT1_PT2_dimgrey
    impute · PR2_PT2_black
    impute · PT1_PT2_lightgrey

Hub definition (unchanged from the earlier analysis so numbers stay comparable):
    |MM| >= 0.70  and  |GS| >= 0.50,   ranked by |MM| x |GS|
      MM = corr(protein, its own module eigengene)      -- centrality in module
      GS = corr(protein, LSR)                           -- link to sweat rate

Statistical note: Enrichr power scales with list size. A module contributing
only ~10 hubs may return few terms, or terms resting on a single gene. The
Overlap column is printed for every term so that can be judged directly.

Outputs (results/wgcna/impute/hub_profile/):
    hub_profile.md          markdown: gene tables + top 5 GO / KEGG per module
    hub_profile.png         figure: hub MM-GS map + top terms per module
    hub_proteins.csv        every hub, with MM, GS, score and official name
    <MODULE>_GO.csv         full significant GO BP result per module
    <MODULE>_KEGG.csv       full significant KEGG result per module

Run with the Python 3.9 interpreter that has PyWGCNA (needs internet):
    python hub_functional_profile.py
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import gseapy as gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from wgcna_evaluation import module_membership, gene_significance
import model_development as md

CASE_DIR = DATA_DIR / "wgcna" / "impute"
OUT = PROJECT_DIR / "results" / "wgcna" / "impute" / "hub_profile"
TRAIT = "LSR (mg/min/cm2)"
MM_HUB, GS_HUB = 0.70, 0.50
FDR = 0.05
TOP_TERMS = 5
TOP_TABLE = 10
GO_LIB, KEGG_LIB = "GO_Biological_Process_2023", "KEGG_2021_Human"

# display name -> (contrast label, exposure codes, moduleColors value)
MODULES = {
    "impute · PT1_PT2_dimgrey":   ("PT1/PT2", ["PT1", "PT2"], "dimgrey"),
    "impute · PR2_PT2_black":     ("PR2/PT2", ["PR2", "PT2"], "black"),
    "impute · PT1_PT2_lightgrey": ("PT1/PT2", ["PT1", "PT2"], "lightgrey"),
}


def build_networks(expr, tr, ex):
    """One PyWGCNA fit per distinct contrast (PT1/PT2 is reused by two modules)."""
    nets = {}
    for _, (contrast, codes, _) in MODULES.items():
        if contrast in nets:
            continue
        print(f"  building {contrast} ...", flush=True)
        mask = pd.Series(ex.isin(codes), index=tr.index)
        with contextlib.redirect_stdout(io.StringIO()):
            nets[contrast] = md.build(expr, tr, mask, contrast.replace("/", "_"))
    return nets


def hubs_of(obj, colour):
    """Hub proteins of one module, ranked by |MM| x |GS|."""
    MM = module_membership(obj, colour)
    GS = gene_significance(obj, TRAIT, MM.index)
    df = pd.DataFrame({"MM": MM, "GS": GS})
    df = df[(df.MM.abs() >= MM_HUB) & (df.GS.abs() >= GS_HUB)]
    df["score"] = df.MM.abs() * df.GS.abs()
    df["symbol"] = [i.split("__")[0] for i in df.index]      # strip bridge suffix
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def gene_names(symbols):
    """Official full names from MyGene.info; blank when a symbol is not found."""
    out, syms = {}, sorted(set(symbols))
    for i in range(0, len(syms), 200):
        r = requests.post("https://mygene.info/v3/query",
                          data={"q": ",".join(syms[i:i + 200]), "scopes": "symbol",
                                "fields": "symbol,name", "species": "human"},
                          timeout=60)
        r.raise_for_status()
        for rec in r.json():
            s, n = rec.get("symbol"), rec.get("name")
            if s and n and s not in out:
                out[s] = n
    return {s: out.get(s, "") for s in syms}


def enrich(genes, library):
    res = gp.enrichr(gene_list=list(genes), gene_sets=[library],
                     organism="human", outdir=None).results
    return res[res["Adjusted P-value"] < FDR].sort_values("Adjusted P-value")


C_GO, C_KEGG, C_WEAK, C_DOT = "#0072B2", "#009E73", "#C0392B", "#D55E00"


def _plot(hub_tables, term_tables):
    """Row 1: where each module's hubs sit in MM-GS space.
       Row 2: their strongest GO and KEGG terms, annotated with the overlap.

    Terms resting on a single gene are drawn in red, because with lists this
    small that is the difference between a finding and an artefact."""
    n = len(MODULES)
    fig = plt.figure(figsize=(6.0 * n, 9.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.35], hspace=0.30, wspace=0.30,
                          top=0.885, bottom=0.07, left=0.055, right=0.98)

    for k, (name, h) in enumerate(hub_tables.items()):
        # ---- MM vs GS -----------------------------------------------------
        ax = fig.add_subplot(gs[0, k])
        ax.scatter(h.MM, h.GS, s=26 + 240 * h.score, color=C_DOT, alpha=.65,
                   edgecolor="white", linewidth=.7, zorder=3)
        for j, r in enumerate(h.head(5).itertuples()):
            off = [(5, 5), (5, -10), (-5, 6), (5, 8), (-5, -11)][j % 5]
            ax.annotate(r.symbol, (r.MM, r.GS), fontsize=8.5, fontweight="bold",
                        xytext=off, textcoords="offset points",
                        ha="right" if off[0] < 0 else "left")
        ax.axvline(MM_HUB, color="#999", ls="--", lw=1)
        ax.axhline(GS_HUB, color="#999", ls="--", lw=1)
        ax.set_xlim(MM_HUB - 0.03, 1.0)
        ax.set_ylim(GS_HUB - 0.03, max(0.9, h.GS.max() + 0.05))
        ax.set_xlabel("Module Membership (MM)", fontsize=9.5)
        if k == 0:
            ax.set_ylabel("Gene Significance vs LSR (GS)", fontsize=9.5)
        ax.set_title(f"{name}\n{len(h)} hub proteins", fontsize=10.5,
                     fontweight="bold")
        ax.grid(color="#EEEEEE", zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

        # ---- top terms ----------------------------------------------------
        ax = fig.add_subplot(gs[1, k])
        go, kegg = term_tables[name]
        rows = ([(r["Term"], r["Adjusted P-value"], r["Overlap"], C_GO)
                 for _, r in go.head(TOP_TERMS).iterrows()] +
                [(r["Term"], r["Adjusted P-value"], r["Overlap"], C_KEGG)
                 for _, r in kegg.head(TOP_TERMS).iterrows()])[::-1]
        if rows:
            v = [-np.log10(f) for _, f, _, _ in rows]
            # a term driven by one gene is coloured red regardless of library
            cols = [C_WEAK if o.split("/")[0] == "1" else c for _, _, o, c in rows]
            ax.barh(range(len(rows)), v, color=cols, zorder=3)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([t[:46] for t, _, _, _ in rows], fontsize=8)
            for i, ((_, _, o, _), val) in enumerate(zip(rows, v)):
                ax.text(val + .05, i, o, va="center", fontsize=7.8, color="#555")
        ax.axvline(-np.log10(FDR), color="#C0392B", ls="--", lw=1.2)
        ax.text(-np.log10(FDR), len(rows) - .35, " FDR 0.05", fontsize=7.5,
                color="#C0392B", va="top")
        ax.set_xlabel("−log10(FDR)     (label = overlap)", fontsize=9)
        ax.set_xlim(0, max(4.2, max(v) * 1.3 if rows else 4.2))
        ax.grid(axis="x", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_GO, C_KEGG, C_WEAK)]
    fig.legend(handles, ["GO Biological Process", "KEGG",
                         "driven by a single gene — do not interpret"],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.008))
    fig.text(0.055, 0.512, "Strongest terms per module — no keyword or topic "
             "filter applied", fontsize=11, fontweight="bold")
    fig.suptitle("Hub proteins of the three modules, and what they enrich for",
                 fontsize=14.5, fontweight="bold", y=0.955)
    f = OUT / "hub_profile.png"
    fig.savefig(f, bbox_inches="tight", dpi=165)
    plt.close(fig)
    print("saved:", f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    print("Rebuilding networks (PyWGCNA):")
    nets = build_networks(expr, tr, ex)

    hub_tables = {name: hubs_of(nets[c], colour)
                  for name, (c, _, colour) in MODULES.items()}
    names = gene_names([s for h in hub_tables.values() for s in h.symbol])

    md_out = ["# Data-driven functional profile of the hub proteins", "",
              f"Hub = |MM| ≥ {MM_HUB} and |GS| ≥ {GS_HUB}, ranked by |MM| × |GS|. "
              f"No keyword list or topic filter is applied anywhere; the terms "
              f"below are simply the lowest-FDR results Enrichr returned "
              f"(FDR < {FDR}).", ""]
    all_rows, term_tables = [], {}

    for name, (contrast, _, colour) in MODULES.items():
        h = hub_tables[name]
        h["name"] = h.symbol.map(names)
        tag = name.split("·")[1].strip()
        all_rows.append(h.assign(module=name))

        shown = h.head(TOP_TABLE)
        md_out += [f"## {name}", "",
                   f"**{len(h)} hub proteins.** "
                   + ("All are listed." if len(h) <= TOP_TABLE
                      else f"Strongest {TOP_TABLE} by MM × GS:"), "",
                   "| # | Symbol | Full gene name | MM | GS | MM × GS |",
                   "|---|---|---|---|---|---|"]
        for i, r in enumerate(shown.itertuples(), 1):
            md_out.append(f"| {i} | **{r.symbol}** | {r.name} | {r.MM:.2f} | "
                          f"{r.GS:.2f} | {r.score:.2f} |")
        md_out.append("")

        for lib, title in ((GO_LIB, "GO Biological Process"), (KEGG_LIB, "KEGG")):
            res = enrich(h.symbol.tolist(), lib)
            term_tables.setdefault(name, []).append(res)
            res.to_csv(OUT / f"{tag}_{'GO' if lib == GO_LIB else 'KEGG'}.csv",
                       index=False)
            md_out += [f"**Top {TOP_TERMS} — {title}**", "",
                       "| Term | Overlap | Adj. p (FDR) |", "|---|---|---|"]
            if len(res):
                for _, r in res.head(TOP_TERMS).iterrows():
                    md_out.append(f"| {r['Term']} | {r['Overlap']} | "
                                  f"{r['Adjusted P-value']:.2e} |")
            else:
                md_out.append(f"| _nothing reached FDR < {FDR}_ | — | — |")
            md_out.append("")
            print(f"  {tag:<20} {title:<24} {len(res):>3} terms at FDR<{FDR}")

    _plot(hub_tables, term_tables)
    pd.concat(all_rows).to_csv(OUT / "hub_proteins.csv", index=False)
    (OUT / "hub_profile.md").write_text("\n".join(md_out))
    print("\nsaved:", OUT / "hub_profile.md")
    print("saved:", OUT / "hub_proteins.csv")


if __name__ == "__main__":
    main()
