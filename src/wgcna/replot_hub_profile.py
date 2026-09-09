"""Redraw hub_profile.png from the CSVs hub_functional_profile.py already wrote.

Rebuilding the WGCNA networks and re-querying Enrichr would risk moving the
numbers (Enrichr's libraries are versioned server-side). This reads the saved
result instead, so the figure can be restyled with no chance of the underlying
values changing.

Needs only matplotlib / pandas / numpy -- no PyWGCNA, no internet:
    python3 src/wgcna/replot_hub_profile.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_profile_plot import plot as plot_hub_profile

OUT = (Path(__file__).resolve().parents[2]
       / "results" / "wgcna" / "impute" / "hub_profile")
MM_HUB, GS_HUB, FDR, TOP_TERMS = 0.70, 0.50, 0.05, 5

# same display names and order as hub_functional_profile.MODULES
MODULES = ["PT1_PT2_dimgrey", "PR2_PT2_black", "PT1_PT2_lightgrey"]


def main():
    hubs = pd.read_csv(OUT / "hub_proteins.csv")

    hub_tables, term_tables = {}, {}
    for tag in MODULES:
        name = f"impute · {tag}"
        h = hubs[hubs.module == name].reset_index(drop=True)
        if h.empty:
            raise SystemExit(f"no hub rows for {name} in hub_proteins.csv")
        hub_tables[name] = h
        term_tables[name] = [pd.read_csv(OUT / f"{tag}_{lib}.csv")
                             for lib in ("GO", "KEGG")]
        print(f"  {tag:<20} {len(h):>3} hubs · "
              f"{len(term_tables[name][0]):>3} GO · "
              f"{len(term_tables[name][1]):>3} KEGG  (FDR<{FDR})")

    plot_hub_profile(hub_tables, term_tables, OUT / "hub_profile.png",
                     MM_HUB, GS_HUB, FDR, TOP_TERMS)


if __name__ == "__main__":
    main()
