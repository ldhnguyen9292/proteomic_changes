"""Complete-case WGCNA — script version of model_development_complete.ipynb.

Mirrors model_development.py but on the 868-protein complete-case set (proteins
with any missing value dropped, no imputation); saves the result figures to
results/wgcna/complete/. Reuses build() and module_trait_heatmap() from
model_development.py so the two scripts stay in sync.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_physiological_data import DATA_DIR, PROJECT_DIR
from wgcna_evaluation import evaluate, scale_free_fit
import model_development as md          # reuse build() + module_trait_heatmap()

CASE_DIR = DATA_DIR / "wgcna" / "complete"
RESULTS_DIR = PROJECT_DIR / "results" / "wgcna" / "complete"
SWEAT_TARGETS = md.SWEAT_TARGETS


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    ex = tr.index.str.split("-").str[-1]

    nets = {
        "full": md.build(expr, tr, None, "full"),
        "PR1/PT1": md.build(expr, tr, pd.Series(ex.isin(["PR1", "PT1"]), index=tr.index), "PR1_PT1"),
        "PR2/PT2": md.build(expr, tr, pd.Series(ex.isin(["PR2", "PT2"]), index=tr.index), "PR2_PT2"),
        "PT1/PT2": md.build(expr, tr, pd.Series(ex.isin(["PT1", "PT2"]), index=tr.index), "PT1_PT2"),
        "PR1/PR2": md.build(expr, tr, pd.Series(ex.isin(["PR1", "PR2"]), index=tr.index), "PR1_PR2"),
    }
    for n, o in nets.items():
        print(f"{n:8s} power={o.power} R²={scale_free_fit(o)[1]:.3f} "
              f"modules={o.datExpr.var['moduleColors'].value_counts().to_dict()}")

    # --- Figure 1: module-trait heatmaps ---
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    axflat = axes.ravel()
    im = None
    for ax, (n, o) in zip(axflat, nets.items()):
        im = md.module_trait_heatmap(ax, o, n)
    for ax in axflat[len(nets):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.5, label="module-trait correlation (r)")
    fig.suptitle("Module–trait correlations, complete-case workflow  (* = BH-FDR < 0.05)",
                 fontsize=13, fontweight="bold")
    f1 = RESULTS_DIR / "module_trait_heatmaps.png"
    fig.savefig(f1, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- Figure 2: evaluation (GS vs MM) ---
    print("\n=== EVALUATION (three criteria) ===")
    fig, axes = plt.subplots(1, len(SWEAT_TARGETS), figsize=(5 * len(SWEAT_TARGETS), 4.6))
    for (net, trait, label), ax in zip(SWEAT_TARGETS, np.atleast_1d(axes)):
        evaluate(nets[net], trait, name=label, ax=ax)
    fig.suptitle("Evaluation — Gene Significance vs Module Membership (complete case)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    f2 = RESULTS_DIR / "evaluation_sweat.png"
    fig.savefig(f2, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("\nsaved:", f1)
    print("saved:", f2)


if __name__ == "__main__":
    main()
