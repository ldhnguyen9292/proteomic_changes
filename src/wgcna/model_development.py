"""Impute-workflow WGCNA — script version of model_development.ipynb.

Builds the four co-expression networks (full + PR1/PT1 + PR2/PT2 + PT1/PT2),
runs the sweat-focused module-trait scan, evaluates the sweat modules against
the three proposal criteria, and SAVES the result figures to
results/wgcna/impute/ (a script cannot display inline like the notebook).

Run with the Python 3.9 interpreter that has PyWGCNA.
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
import PyWGCNA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # wgcna_evaluation
from read_physiological_data import DATA_DIR, PROJECT_DIR
from wgcna_evaluation import evaluate, module_trait_table, scale_free_fit

CASE_DIR = DATA_DIR / "wgcna" / "impute"
RESULTS_DIR = PROJECT_DIR / "results" / "wgcna" / "impute"
# Sweat-association targets. Local sweat rate (LSR) is the sweat measure that
# varies per condition, so every contrast is evaluated against LSR for a single,
# consistent sweat trait (matching hub_prioritization.py and enrichment.py).
SWEAT_TARGETS = [
    ("PR1/PT1", "LSR (mg/min/cm2)", "PR1/PT1 · LSR  (heat stress, pre)"),
    ("PR2/PT2", "LSR (mg/min/cm2)", "PR2/PT2 · LSR  (heat stress, post)"),
    ("PT1/PT2", "LSR (mg/min/cm2)", "PT1/PT2 · LSR  (acclimation, heat)"),
    ("PR1/PR2", "LSR (mg/min/cm2)", "PR1/PR2 · LSR  (acclimation, rest)"),
]


def build(expr, tr, mask, name):
    e, t = (expr, tr) if mask is None else (expr.loc[mask], tr.loc[mask])
    o = PyWGCNA.WGCNA(name=name, species="human", geneExp=e, sampleInfo=t,
                      TPMcutoff=float("-inf"), RsquaredCut=0.8,
                      networkType="signed hybrid", TOMType="signed",
                      minModuleSize=30, save=False)
    o.runWGCNA()
    return o


def module_trait_heatmap(ax, obj, name):
    """Modules (rows) x traits (cols) coloured by r; '*' marks BH-FDR < 0.05."""
    traits = [t for t in obj.datExpr.obs.columns if obj.datExpr.obs[t].nunique() > 1]
    mods = [m[2:] for m in obj.MEs.columns]
    R = pd.DataFrame(index=mods, columns=traits, dtype=float)
    sig = pd.DataFrame("", index=mods, columns=traits)
    for t in traits:
        mt = module_trait_table(obj, t)
        for m in mods:
            R.loc[m, t] = mt.loc[m, "r"]
            if mt.loc[m, "FDR"] < 0.05:
                sig.loc[m, t] = "*"
    im = ax.imshow(R.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([t.split(" (")[0] for t in traits], rotation=90, fontsize=6)
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(mods, fontsize=7)
    for i in range(len(mods)):
        for j in range(len(traits)):
            ax.text(j, i, f"{R.iloc[i, j]:.2f}{sig.iloc[i, j]}",
                    ha="center", va="center", fontsize=5)
    ax.set_title(f"{name}  (power {obj.power}, R²={scale_free_fit(obj)[1]:.2f})", fontsize=9)
    return im


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    nets = {
        "full": build(expr, tr, None, "full"),
        "PR1/PT1": build(expr, tr, pd.Series(ex.isin(["PR1", "PT1"]), index=tr.index), "PR1_PT1"),
        "PR2/PT2": build(expr, tr, pd.Series(ex.isin(["PR2", "PT2"]), index=tr.index), "PR2_PT2"),
        "PT1/PT2": build(expr, tr, pd.Series(ex.isin(["PT1", "PT2"]), index=tr.index), "PT1_PT2"),
        "PR1/PR2": build(expr, tr, pd.Series(ex.isin(["PR1", "PR2"]), index=tr.index), "PR1_PR2"),
    }
    for n, o in nets.items():
        print(f"{n:8s} power={o.power} R²={scale_free_fit(o)[1]:.3f} "
              f"modules={o.datExpr.var['moduleColors'].value_counts().to_dict()}")

    # --- Figure 1: module-trait heatmaps (all networks) ---
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    axflat = axes.ravel()
    im = None
    for ax, (n, o) in zip(axflat, nets.items()):
        im = module_trait_heatmap(ax, o, n)
    for ax in axflat[len(nets):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.5, label="module-trait correlation (r)")
    fig.suptitle("Module–trait correlations, impute workflow  (* = BH-FDR < 0.05)",
                 fontsize=13, fontweight="bold")
    f1 = RESULTS_DIR / "module_trait_heatmaps.png"
    fig.savefig(f1, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- Figure 2: evaluation (GS vs MM) for the sweat modules ---
    print("\n=== EVALUATION (three criteria) ===")
    fig, axes = plt.subplots(1, len(SWEAT_TARGETS), figsize=(5 * len(SWEAT_TARGETS), 4.6))
    for (net, trait, label), ax in zip(SWEAT_TARGETS, np.atleast_1d(axes)):
        evaluate(nets[net], trait, name=label, ax=ax)
    fig.suptitle("Evaluation — Gene Significance vs Module Membership (impute)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    f2 = RESULTS_DIR / "evaluation_sweat.png"
    fig.savefig(f2, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("\nsaved:", f1)
    print("saved:", f2)


if __name__ == "__main__":
    main()
