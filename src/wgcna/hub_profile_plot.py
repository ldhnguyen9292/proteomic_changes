"""The hub-profile figure, separated from the analysis that produces it.

Kept apart from hub_functional_profile.py so the figure can be redrawn from the
saved CSVs (see replot_hub_profile.py) without rebuilding the WGCNA networks or
re-querying Enrichr -- redrawing must never be able to move the numbers.

Only matplotlib / pandas / numpy are needed here.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_GO, C_KEGG, C_WEAK, C_DOT = "#0072B2", "#009E73", "#C0392B", "#D55E00"


def _shown_of(n_sig, top_terms, label):
    """'top 5 of 24 GO' / 'all 4 KEGG' / 'no significant KEGG'."""
    if n_sig == 0:
        return f"no significant {label}"
    if n_sig <= top_terms:
        return f"all {n_sig} {label}"
    return f"top {top_terms} of {n_sig} {label}"


def plot(hub_tables, term_tables, out_file, mm_hub=0.70, gs_hub=0.50,
         fdr=0.05, top_terms=5):
    """Row 1: where each module's hubs sit in MM-GS space.
       Row 2: their strongest GO and KEGG terms, annotated with the overlap.

    Terms resting on a single gene are drawn in red, because with lists this
    small that is the difference between a finding and an artefact.

    hub_tables:  {module name -> DataFrame with MM, GS, score, symbol}
    term_tables: {module name -> [full significant GO df, full significant KEGG df]}
                 Full, not truncated -- the panel title reports how many of the
                 significant terms the panel is actually showing.
    """
    n = len(hub_tables)
    fig = plt.figure(figsize=(6.0 * n, 9.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.35], hspace=0.42, wspace=0.42,
                          top=0.885, bottom=0.105, left=0.055, right=0.98)

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
        ax.axvline(mm_hub, color="#999", ls="--", lw=1)
        ax.axhline(gs_hub, color="#999", ls="--", lw=1)
        ax.set_xlim(mm_hub - 0.03, 1.0)
        ax.set_ylim(gs_hub - 0.03, max(0.9, h.GS.max() + 0.05))
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
                 for _, r in go.head(top_terms).iterrows()] +
                [(r["Term"], r["Adjusted P-value"], r["Overlap"], C_KEGG)
                 for _, r in kegg.head(top_terms).iterrows()])[::-1]
        if rows:
            v = [-np.log10(f) for _, f, _, _ in rows]
            # a term driven by one gene is coloured red regardless of library
            cols = [C_WEAK if o.split("/")[0] == "1" else c for _, _, o, c in rows]
            ax.barh(range(len(rows)), v, color=cols, zorder=3)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([t[:46] for t, _, _, _ in rows], fontsize=8)
            for i, ((_, _, o, _), val) in enumerate(zip(rows, v)):
                ax.text(val + .05, i, o, va="center", fontsize=7.8, color="#555")
        ax.axvline(-np.log10(fdr), color="#C0392B", ls="--", lw=1.2)
        ax.text(-np.log10(fdr), len(rows) - .35, " FDR 0.05", fontsize=7.5,
                color="#C0392B", va="top")
        ax.set_xlabel("−log10(FDR)     (label = overlap)", fontsize=9)
        ax.set_xlim(0, max(4.2, max(v) * 1.3 if rows else 4.2))
        ax.grid(axis="x", color="#EEEEEE", zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

        # how much of the significant result this panel is actually showing
        ax.set_title(f"{_shown_of(len(go), top_terms, 'GO')}  ·  "
                     f"{_shown_of(len(kegg), top_terms, 'KEGG')}"
                     f"   significant at FDR<{fdr}",
                     fontsize=9, color="#333", pad=16)

        # a panel whose displayed terms share one FDR is not a ranking
        tied = [(res, lab) for res, lab in ((go, "GO"), (kegg, "KEGG"))
                if len(res) > top_terms
                and res.head(top_terms)["Adjusted P-value"].nunique() == 1]
        if tied:
            note = "; ".join(
                f"all {len(res)} {lab} terms tie at FDR "
                f"{res['Adjusted P-value'].iloc[0]:.2g} — which {top_terms} "
                f"are shown is arbitrary" for res, lab in tied)
            ax.annotate(note, xy=(0, 1), xycoords="axes fraction",
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=7.4, color=C_WEAK, fontweight="bold", va="bottom")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_GO, C_KEGG, C_WEAK)]
    fig.legend(handles, ["GO Biological Process", "KEGG",
                         "driven by a single gene — do not interpret"],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.008))
    fig.text(0.055, 0.532, "Strongest terms per module — no keyword or topic "
             "filter applied", fontsize=11, fontweight="bold")
    fig.text(0.055, 0.514, "Bars are the lowest-FDR terms per library, not the "
             "full significant result; each panel states how many of its "
             "significant terms are shown.", fontsize=8.8, color="#444")
    fig.suptitle("Hub proteins of the three modules, and what they enrich for",
                 fontsize=14.5, fontweight="bold", y=0.955)
    fig.savefig(out_file, bbox_inches="tight", dpi=165)
    plt.close(fig)
    print("saved:", out_file)
