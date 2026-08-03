"""Unbiased functional profile of the hub proteins, with no theme filtering.

Deliberately contains no keyword or topic list. Enrichr is queried with each
module's hub proteins and whatever comes back at the lowest FDR is reported, so
nothing is pre-selected by the analyst.

Official gene names come from MyGene.info rather than being written by hand.

Hub set = every protein passing |MM| >= 0.70 and |GS| >= 0.50 in that module
(the hub_prioritization.csv list). The summary table shows the strongest ones
by MM x GS; the enrichment uses the full hub list for that module, because
splitting it further would cost power that these list sizes cannot spare.

Outputs (results/wgcna/impute/hub_enrichment/):
  hub_profile_unbiased.md    ready-to-paste markdown (tables + top 5 GO / KEGG)
  hub_profile_top.csv        the hub table with official gene names

Usage (Python 3.8+, needs internet):
  python hub_profile_unbiased.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import gseapy as gp
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import PROJECT_DIR

RES = PROJECT_DIR / "results" / "wgcna" / "impute"
OUT = RES / "hub_enrichment"
GO_LIB, KEGG_LIB = "GO_Biological_Process_2023", "KEGG_2021_Human"
FDR = 0.05
TOP_TABLE = 10          # how many hubs to list per module in the summary table
TOP_TERMS = 5           # top terms per library, per module

# full module name -> (report label, moduleColors value in hub_prioritization.csv)
MODULES = {
    "impute · PT1_PT2_dimgrey":   ("M1", "dimgrey"),
    "impute · PR2_PT2_black":     ("M2", "black"),
    "impute · PT1_PT2_lightgrey": ("M3", "lightgrey"),
}


def gene_names(symbols):
    """Official full names from MyGene.info; blank if the symbol is not found."""
    out = {}
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 200):
        chunk = syms[i:i + 200]
        r = requests.post("https://mygene.info/v3/query",
                          data={"q": ",".join(chunk), "scopes": "symbol",
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hubs = pd.read_csv(RES / "hub_prioritization.csv")
    names = gene_names([g.split("__")[0] for g in hubs.protein])

    md = ["# Unbiased functional profile of the hub proteins", "",
          "No keyword list, no target topics. Terms are whatever Enrichr returns "
          f"at the lowest FDR (< {FDR}).", ""]
    table_rows = []

    for full_name, (label, colour) in MODULES.items():
        sub = hubs[hubs.module == colour].copy()
        sub["symbol"] = sub.protein.str.split("__").str[0]
        genes = sub.symbol.tolist()
        top = sub.head(TOP_TABLE)

        md += [f"## {full_name}  ({label})", "",
               f"**{len(genes)} hub proteins** "
               f"(|MM| ≥ 0.70 and |GS| ≥ 0.50). "
               f"{'All are listed.' if len(genes) <= TOP_TABLE else f'Strongest {TOP_TABLE} by MM × GS:'}",
               "", "| # | Symbol | Full gene name | MM | GS | MM × GS |",
               "|---|---|---|---|---|---|"]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            md.append(f"| {i} | **{r.symbol}** | {names.get(r.symbol,'')} | "
                      f"{r.MM:.2f} | {r.GS:.2f} | {r.score:.2f} |")
            table_rows.append(dict(module_full=full_name, label=label, rank=i,
                                   symbol=r.symbol, name=names.get(r.symbol, ""),
                                   MM=r.MM, GS=r.GS, score=r.score))
        md.append("")

        for lib, title in ((GO_LIB, "GO Biological Process"), (KEGG_LIB, "KEGG")):
            res = enrich(genes, lib)
            md += [f"**Top {TOP_TERMS} — {title}**", "",
                   "| Term | Overlap | Adj. p (FDR) |", "|---|---|---|"]
            if len(res):
                for _, r in res.head(TOP_TERMS).iterrows():
                    md.append(f"| {r.Term} | {r.Overlap} | {r['Adjusted P-value']:.2e} |")
            else:
                md.append(f"| _no term reached FDR < {FDR}_ | — | — |")
            md.append("")
            print(f"{label:>3} {title:<24} {len(res):>3} terms at FDR<{FDR}")

    pd.DataFrame(table_rows).to_csv(OUT / "hub_profile_top.csv", index=False)
    (OUT / "hub_profile_unbiased.md").write_text("\n".join(md))
    print("\nsaved:", OUT / "hub_profile_unbiased.md")
    print("saved:", OUT / "hub_profile_top.csv")


if __name__ == "__main__":
    main()
