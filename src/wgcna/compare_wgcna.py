"""Compare the impute vs complete-case WGCNA workflows for sweat.

Fits both protein sets (KNN-imputed 1707 vs complete-case 868) across the full,
PR1/PT1 and PR2/PT2 networks, and tabulates -- side by side -- the strongest
sweat module (module eigengene vs trait) and its top hub protein (Gene
Significance). Focus traits: whole-body sweat rate and LSR.

Output: results/wgcna/impute_vs_complete.png  (+ printed table)
Run with the Python 3.9 kernel that has PyWGCNA.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import PyWGCNA

# shared ingestion modules live in src/data_pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR, PROJECT_DIR

RESULTS_DIR = PROJECT_DIR / "results" / "wgcna"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CASES = {
    "Impute (1707)": (DATA_DIR / "wgcna" / "impute" / "wgcna_expression.csv",
                      DATA_DIR / "wgcna" / "impute" / "wgcna_traits.csv"),
    "Complete (868)": (DATA_DIR / "wgcna" / "complete" / "wgcna_expression.csv",
                       DATA_DIR / "wgcna" / "complete" / "wgcna_traits.csv"),
}
SWEAT = ["Sweat rate (L/h)", "LSR (mg/min/cm2)"]
INK, GRID = "#222222", "#DDDDDD"


def _corr(a, b):
    k = ~(pd.isna(a) | pd.isna(b))
    return pearsonr(a[k], b[k])


def fit(expr, tr, mask):
    e = expr.loc[mask] if mask is not None else expr
    t = tr.loc[mask] if mask is not None else tr
    o = PyWGCNA.WGCNA(name='x', species='human', geneExp=e, sampleInfo=t,
                      TPMcutoff=float('-inf'), RsquaredCut=0.8,
                      networkType='signed hybrid', TOMType='signed',
                      minModuleSize=30, save=False)
    o.runWGCNA()
    return o


def best_module(o, trait):
    """Strongest module for the trait + its top hub protein by |GS|."""
    ME = o.MEs
    y = o.datExpr.obs.loc[ME.index, trait].astype(float).values
    mt = pd.DataFrame({m[2:]: _corr(ME[m].values, y) for m in ME.columns},
                      index=['r', 'p']).T
    mt = mt.reindex(mt['r'].abs().sort_values(ascending=False).index)
    mod, r, p = mt.index[0], mt.iloc[0]['r'], mt.iloc[0]['p']
    X = o.datExpr.to_df()
    genes = o.datExpr.var.index[o.datExpr.var['moduleColors'] == mod]
    gs = pd.Series({g: _corr(X[g].values, y)[0] for g in genes}).sort_values(
        key=abs, ascending=False)
    return dict(module=mod, r=r, p=p, n7=int((gs.abs() > 0.7).sum()),
                top=gs.index[0], top_gs=gs.iloc[0])


def main():
    records = []
    for case, (ep, tp) in CASES.items():
        expr = pd.read_csv(ep, index_col=0)
        tr = pd.read_csv(tp, index_col=0)
        common = expr.index.intersection(tr.index)
        expr, tr = expr.loc[common], tr.loc[common]
        ex = tr.index.str.split('-').str[-1]
        nets = {'full': None,
                'PR1/PT1': pd.Series(ex.isin(['PR1', 'PT1']), index=tr.index),
                'PR2/PT2': pd.Series(ex.isin(['PR2', 'PT2']), index=tr.index),
                'PT1/PT2': pd.Series(ex.isin(['PT1', 'PT2']), index=tr.index)}
        for net, mask in nets.items():
            o = fit(expr, tr, mask)
            nmod = o.MEs.shape[1]
            for trait in SWEAT:
                b = best_module(o, trait)
                records.append({'case': case, 'nprot': expr.shape[1], 'network': net,
                                'nmod': nmod, 'trait': trait, **b})
    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    _plot(df)


def _plot(df):
    cases = list(CASES.keys())
    keys = [(net, tr) for net in ['full', 'PR1/PT1', 'PR2/PT2', 'PT1/PT2'] for tr in SWEAT]

    def fmt(row):
        star = " *" if abs(row['r']) > 0.6 and row['p'] < 0.05 else ""
        return (f"{row['module']}  r={row['r']:+.2f} (p={row['p']:.3f}){star}\n"
                f"|GS|>0.7: {row['n7']}   top: {row['top']} ({row['top_gs']:+.2f})")

    col_labels = ["Network / trait"] + cases
    cell_text = []
    for net, tr in keys:
        rowcells = [f"{net}\n{tr.split(' (')[0]}"]
        for case in cases:
            r = df[(df.case == case) & (df.network == net) & (df.trait == tr)].iloc[0]
            rowcells.append(fmt(r))
        cell_text.append(rowcells)

    row_lines = [1] + [3] * len(keys)
    unit = 0.92 / sum(row_lines)
    fig, ax = plt.subplots(figsize=(13, 0.5 * sum(row_lines) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   colWidths=[0.18, 0.41, 0.41], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(1.2)
        cell.set_height(row_lines[rr] * unit)
        cell.set_text_props(va="center")
        if rr == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", fontweight="bold", va="center")
        else:
            cell.set_facecolor("#F5F5F5" if rr % 2 else "#FFFFFF")
            if cc == 0:
                cell.set_text_props(fontweight="bold", va="center")

    np_i = df[df.case == cases[0]]['nprot'].iloc[0]
    np_c = df[df.case == cases[1]]['nprot'].iloc[0]
    ax.set_title("WGCNA workflows compared — strongest sweat module & top hub\n"
                 f"Impute = {np_i} proteins (KNN) · Complete = {np_c} proteins "
                 f"(drop any missing) · * = module |r|>0.6 & p<0.05",
                 fontsize=12, fontweight="bold", pad=16)
    fig.tight_layout()
    out = RESULTS_DIR / "impute_vs_complete.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("\nSaved figure:", out)


if __name__ == "__main__":
    main()
