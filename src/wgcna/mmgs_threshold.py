"""MM-GS (criterion 3) per module + threshold sensitivity.

Answers "where should the MM-GS cut-off sit, and does it matter?" for the chosen
impute 10% / LSR analysis.

MM-GS is the correlation, ACROSS THE PROTEINS INSIDE ONE MODULE, between Module
Membership (kME) and Gene Significance (correlation with LSR). It is a different
quantity from the module-trait correlation shown in the module-trait heatmap --
that one correlates the module EIGENGENE with the trait across samples. A module
can have a strong eigengene-trait correlation and a near-zero MM-GS.

Note on p-values: MM-GS is tested over n = number of proteins in the module
(127-946 here), and those proteins are co-expressed by construction, so the test
is anti-conservative twice over. PT1/PT2 darkgrey reaches p = 0.011 at r = 0.162.
MM-GS should be judged on effect size, not significance -- hence the threshold
question.

Reads the per-module flags written by enrichment.py (no WGCNA rebuild needed):
  results/wgcna/{case}/enrichment/module_selection.csv

Usage (any Python 3.8+ with pandas/matplotlib):
  python mmgs_threshold.py            # impute (primary)
  python mmgs_threshold.py complete   # complete-case sensitivity
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import PROJECT_DIR

PROPOSAL_CUT, RELAXED_CUT = 0.6, 0.5
PASS_C, FAIL_C, OTHER_C = "#0072B2", "#D55E00", "#BBBBBB"


def load(case):
    f = PROJECT_DIR / "results" / "wgcna" / case / "enrichment" / "module_selection.csv"
    if not f.exists():
        sys.exit(f"missing {f}\nrun:  python enrichment.py {case}")
    return pd.read_csv(f)


def plateau(vals, at=PROPOSAL_CUT, lo=0.0, hi=1.0):
    """Cut-off range CONTAINING `at` over which the pass-count is constant.

    Deliberately not the globally widest gap: the decision-relevant question is
    how far the cut-off can move from the one actually used before the answer
    changes. For the complete case the widest gap sits below both candidate
    cut-offs and would overstate the stability.
    """
    edges = np.concatenate([[lo], np.sort(np.abs(vals)), [hi]])
    i = int(np.searchsorted(edges, at, side="right")) - 1
    return edges[i], edges[i + 1]


def main(case="impute"):
    df = load(case)
    # Candidates = modules that already cleared criteria 1-2. Criterion 3 is what
    # this figure is about, so it must NOT be pre-applied here -- otherwise the
    # module it excludes (darkgrey) would vanish and the plot would be circular.
    focus = df[df["passes_topology"] & df["passes_fdr"] & df["passes_mt_r"]].copy()
    focus["label"] = focus["network"] + " · " + focus["module"]
    focus["absmm"] = focus["mmgs_r"].abs()
    focus = focus.sort_values("absmm")

    sel = focus
    lo, hi = plateau(sel["mmgs_r"].abs().values)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(14.5, 0.62 * len(focus) + 4.2),
        gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- Panel A: MM-GS per module -------------------------------------------
    others = df[~df.index.isin(focus.index)]["mmgs_r"].abs().values
    axA.scatter(others, np.full(len(others), -0.85), s=22, color=OTHER_C,
                alpha=0.55, linewidth=0, zorder=1)
    axA.text(1.005, -0.85, "  modules failing\n  criteria 1-2", fontsize=7,
             color="#777", va="center")

    y = np.arange(len(focus))
    colors = [PASS_C if a >= PROPOSAL_CUT else FAIL_C for a in focus["absmm"]]
    axA.hlines(y, 0, focus["absmm"], color=colors, linewidth=2.4, zorder=2)
    axA.scatter(focus["absmm"], y, color=colors, s=95, zorder=3,
                edgecolor="white", linewidth=0.9)
    for yi, (_, r) in zip(y, focus.iterrows()):
        axA.text(r["absmm"] + 0.022, yi, f"{r['mmgs_r']:+.3f}", va="center",
                 fontsize=8.5, fontweight="bold",
                 color=PASS_C if r["absmm"] >= PROPOSAL_CUT else FAIL_C)

    labels = [f"{r['label']}\n"
              f"n={r['n_prot']}, module r={r['mt_r']:+.2f}, FDR={r['mt_FDR']:.3f}"
              for _, r in focus.iterrows()]
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=8)
    axA.set_ylim(-1.6, len(focus) - 0.3)

    axA.axvspan(lo, hi, color="#4DAF4A", alpha=0.11, zorder=0)
    axA.axvline(PROPOSAL_CUT, color="#333", ls="--", lw=1.5, zorder=4)
    axA.axvline(RELAXED_CUT, color="#888", ls=":", lw=1.4, zorder=4)
    axA.text(PROPOSAL_CUT, len(focus) - 0.42, " proposal 0.6", fontsize=8.5,
             fontweight="bold", color="#333", ha="left")
    axA.text(RELAXED_CUT, len(focus) - 0.42, "relaxed 0.5 ", fontsize=8,
             color="#888", ha="right")
    axA.set_xlim(0, 1.0)
    axA.set_xlabel("|MM-GS|  —  Module Membership vs Gene Significance (LSR)")
    axA.set_title(f"A. Criterion 3 per module — {case}\n"
                  "modules that already cleared criteria 1-2; "
                  "green band = cut-offs giving the same answer",
                  fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.grid(axis="x", color="#EEEEEE", zorder=0)
    axA.set_axisbelow(True)

    # ---- Panel B: threshold sensitivity --------------------------------------
    grid = np.linspace(0, 1, 501)
    counts = [(sel["mmgs_r"].abs() >= t).sum() for t in grid]
    axB.step(grid, counts, where="post", color="#333", lw=2)
    axB.axvspan(lo, hi, color="#4DAF4A", alpha=0.11)
    axB.axvline(PROPOSAL_CUT, color="#333", ls="--", lw=1.5)
    axB.axvline(RELAXED_CUT, color="#888", ls=":", lw=1.4)
    n_at = int((sel["mmgs_r"].abs() >= PROPOSAL_CUT).sum())
    axB.annotate(f"at 0.6 → {n_at} of {len(sel)} modules",
                 xy=(PROPOSAL_CUT, n_at), xytext=(PROPOSAL_CUT - 0.30, n_at + 0.75),
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.1))
    axB.text((lo + hi) / 2, 0.30, f"stable\n{lo:.2f} – {hi:.2f}", ha="center",
             fontsize=8.5, color="#2E7D32", fontweight="bold")
    axB.set_xlim(0, 1)
    axB.set_ylim(0, len(sel) + 1.4)
    axB.set_yticks(range(len(sel) + 1))
    axB.set_xlabel("MM-GS cut-off")
    axB.set_ylabel("criterion 1-2 modules also passing criterion 3")
    axB.set_title("B. Threshold sensitivity\n"
                  "flat = the cut-off does not change the result",
                  fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.grid(color="#EEEEEE")
    axB.set_axisbelow(True)

    equiv = ("0.5 and 0.6 are equivalent here"
             if lo <= RELAXED_CUT and hi >= PROPOSAL_CUT
             else f"note: 0.5 and 0.6 do NOT agree — 0.5 keeps "
                  f"{int((sel['mmgs_r'].abs() >= RELAXED_CUT).sum())}, "
                  f"0.6 keeps {n_at}")
    fig.suptitle(
        f"MM-GS (criterion 3) and its threshold — {case}, LSR\n"
        f"any cut-off in {lo:.2f}–{hi:.2f} selects the same modules; {equiv}",
        fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()

    out = PROJECT_DIR / "results" / "wgcna" / f"mmgs_threshold_{case}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    tab = focus[["network", "module", "n_prot", "mt_r", "mt_FDR", "mmgs_r",
                 "mmgs_p", "selected"]].sort_values("mmgs_r", ascending=False)
    tab.to_csv(PROJECT_DIR / "results" / "wgcna" / f"mmgs_by_module_{case}.csv",
               index=False)
    print(tab.to_string(index=False))
    print(f"\nplateau: any MM-GS cut-off in {lo:.3f}–{hi:.3f} gives the same "
          f"{int((sel['mmgs_r'].abs() >= hi).sum())} of {len(sel)} modules")
    print("saved:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "impute")
