"""Gene-level redundancy behind the terms plotted in the hub-profile figure.

Reads the enrichment tables written by hub_functional_profile.py and draws, for
each module, a gene x term membership matrix over exactly the terms that appear
in the bottom row of hub_profile.png (top 5 GO + top 5 KEGG).

The point it answers: the bars are not independent findings. Most of them are
the same handful of hub proteins re-counted under different labels.

Output: results/wgcna/impute/hub_profile/hub_term_gene_overlap.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results" / "wgcna" / "impute" / "hub_profile"
MODULES = ["PT1_PT2_dimgrey", "PR2_PT2_black", "PT1_PT2_lightgrey"]
TOP_TERMS = 5

C_GO, C_KEGG, C_WEAK = "#0072B2", "#009E73", "#C0392B"


def plotted_terms(module):
    """The exact terms drawn in hub_profile.png, in the same order."""
    rows = []
    for lib, colour in ((("GO"), C_GO), ("KEGG", C_KEGG)):
        df = pd.read_csv(OUT / f"{module}_{lib}.csv").head(TOP_TERMS)
        for _, r in df.iterrows():
            single = r["Overlap"].split("/")[0] == "1"
            rows.append({"lib": lib, "term": r["Term"], "overlap": r["Overlap"],
                         "genes": r["Genes"].split(";"),
                         "colour": C_WEAK if single else colour})
    return rows


def main():
    data = {m: plotted_terms(m) for m in MODULES}
    # genes ordered by how many of the plotted terms they appear in
    order = {}
    for m, rows in data.items():
        counts = pd.Series([g for r in rows for g in r["genes"]]).value_counts()
        order[m] = list(counts.sort_values(ascending=False).index)

    # the widest module gets its own row; the two small ones share the row below
    wide, small = "PR2_PT2_black", ["PT1_PT2_dimgrey", "PT1_PT2_lightgrey"]
    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[len(order[m]) for m in small],
                          height_ratios=[1, 1], hspace=1.05, wspace=0.55,
                          top=0.845, bottom=0.115, left=0.135, right=0.99)
    axes = {wide: fig.add_subplot(gs[0, :]),
            small[0]: fig.add_subplot(gs[1, 0]),
            small[1]: fig.add_subplot(gs[1, 1])}

    for m in MODULES:
        rows, genes = data[m], order[m]
        ax = axes[m]
        ny = len(rows)

        for i, r in enumerate(rows):
            y = ny - 1 - i
            for j, g in enumerate(genes):
                if g in r["genes"]:
                    ax.scatter(j, y, s=96, color=r["colour"], zorder=3)
            # connect the genes of one term so re-use reads as a horizontal run
            xs = [j for j, g in enumerate(genes) if g in r["genes"]]
            if len(xs) > 1:
                ax.plot([min(xs), max(xs)], [y, y], color=r["colour"], lw=1.4,
                        alpha=.35, zorder=2)

        reused = {g for g in genes
                  if sum(g in r["genes"] for r in rows) > 1}
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=90, fontsize=7.5)
        for lab in ax.get_xticklabels():
            if lab.get_text() in reused:
                lab.set_fontweight("bold")
        ax.set_yticks(range(ny))
        ax.set_yticklabels([f"{r['term'][:44]}  [{r['overlap']}]"
                            for r in rows][::-1], fontsize=7.5)
        ax.set_xlim(-0.7, len(genes) - 0.3)
        ax.set_ylim(-0.7, ny - 0.3)
        ax.grid(color="#EFEFEF", lw=.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        slots = sum(len(r["genes"]) for r in rows)
        ax.set_title(f"impute · {m}\n{ny} terms · {len(genes)} distinct genes · "
                     f"{slots} gene-slots · {len(reused)} genes reused",
                     fontsize=9.5, fontweight="bold")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8)
               for c in (C_GO, C_KEGG, C_WEAK)]
    fig.legend(handles, ["GO Biological Process", "KEGG",
                         "term rests on a single gene — do not interpret"],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.008))
    fig.suptitle("The same hub proteins carry most of the enriched terms",
                 fontsize=14, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925, "Each row is a bar from the bottom of hub_profile.png; "
             "a dot means that protein is one of the genes counted in that "
             "term's overlap. Bold gene = counted in more than one term.",
             fontsize=9.5, ha="center", color="#444")

    f = OUT / "hub_term_gene_overlap.png"
    fig.savefig(f, bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("saved:", f)


if __name__ == "__main__":
    main()
