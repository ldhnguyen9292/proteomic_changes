"""WGCNA evaluation following the proposal's three criteria.

  1. Network topology   -- scale-free fit R^2 at the selected soft-power (> 0.8).
  2. Module-trait        -- Pearson r + p + BH-FDR between module eigengenes
                            and a trait.
  3. Biomarker priority  -- correlation of Module Membership (MM = kME) vs Gene
                            Significance (GS) within a target module
                            (|r| > 0.6 & p < 0.05); hubs = high |MM| and |GS|.

Operates on an already-fitted PyWGCNA object (does not import PyWGCNA).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def _bh(p):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.clip(ranked, 0, 1)
    return out


def _corr(a, b):
    k = ~(pd.isna(a) | pd.isna(b))
    return pearsonr(a[k], b[k])


def scale_free_fit(obj):
    """Criterion 1 -> (selected power, scale-free R^2)."""
    row = obj.sft[obj.sft["Power"] == obj.power]
    return int(obj.power), float(row["SFT.R.sq"].iloc[0])


def module_trait_table(obj, trait):
    """Criterion 2 -> module eigengene vs trait: r, p, BH-FDR (sorted by |r|)."""
    ME = obj.MEs
    y = obj.datExpr.obs.loc[ME.index, trait].astype(float).values
    df = pd.DataFrame({m[2:]: _corr(ME[m].values, y) for m in ME.columns},
                      index=["r", "p"]).T
    df["FDR"] = _bh(df["p"].values)
    return df.sort_values("r", key=abs, ascending=False)


def module_membership(obj, module):
    """MM / kME: correlation of each module protein with the module eigengene."""
    X = obj.datExpr.to_df()
    me = obj.MEs["ME" + module].values
    genes = obj.datExpr.var.index[obj.datExpr.var["moduleColors"] == module]
    return pd.Series({g: _corr(X[g].values, me)[0] for g in genes})


def gene_significance(obj, trait, genes):
    """GS: correlation of each protein with the trait."""
    X = obj.datExpr.to_df()
    y = obj.datExpr.obs.loc[X.index, trait].astype(float).values
    return pd.Series({g: _corr(X[g].values, y)[0] for g in genes})


def evaluate(obj, trait, module=None, name=None, ax=None,
             r2_cut=0.8, mt_cut=0.5, mmgs_cut=0.6, mm_hub=0.8, gs_hub=0.7):
    """Evaluate one target module against all three criteria + GS-vs-MM scatter.

    module=None auto-selects the strongest module-trait module (robust across
    networks whose module colours differ). Returns a summary dict; the returned
    'hubs' are proteins with |MM| > mm_hub and |GS| > gs_hub (high MM AND GS).
    """
    power, r2 = scale_free_fit(obj)
    mt = module_trait_table(obj, trait)
    if module is None:
        module = mt.index[0]
    r, p, fdr = mt.loc[module, ["r", "p", "FDR"]]

    MM = module_membership(obj, module)
    GS = gene_significance(obj, trait, MM.index)
    mmgs_r, mmgs_p = pearsonr(MM.values, GS.values)

    hubs = pd.DataFrame({"MM": MM, "GS": GS})
    hubs = (hubs[(hubs.MM.abs() > mm_hub) & (hubs.GS.abs() > gs_hub)]
            .assign(score=lambda d: d.MM.abs() * d.GS.abs())
            .sort_values("score", ascending=False))

    tag = name or f"{module} / {trait.split(' (')[0]}"
    ok = lambda b: "PASS" if b else "----"
    print(f"=== {tag}  (module '{module}') ===")
    print(f"  [1] topology    : power={power}, R^2={r2:.3f}   "
          f"{ok(r2 > r2_cut)} (>{r2_cut})")
    print(f"  [2] module-trait: r={r:+.3f}, p={p:.3g}, FDR={fdr:.3g}   "
          f"{ok(abs(r) > mt_cut and fdr < 0.05)} (|r|>{mt_cut} & FDR<0.05)")
    print(f"  [3] MM vs GS    : r={mmgs_r:+.3f}, p={mmgs_p:.2g}   "
          f"{ok(abs(mmgs_r) > mmgs_cut and mmgs_p < 0.05)} (|r|>{mmgs_cut} & p<0.05)")
    print(f"      hubs (|MM|>{mm_hub} & |GS|>{gs_hub}): {list(hubs.index) or 'none'}")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(MM.values, GS.values, s=14, color="#BBBBBB", alpha=0.6, linewidth=0)
    if len(hubs):
        ax.scatter(hubs.MM, hubs.GS, s=42, color="#D55E00",
                   edgecolor="white", linewidth=0.5, zorder=3)
        for g, row in hubs.head(8).iterrows():
            ax.annotate(g, (row.MM, row.GS), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    for x in (mm_hub, -mm_hub):
        ax.axvline(x, color="#CC3311", ls="--", lw=0.8)
    for y in (gs_hub, -gs_hub):
        ax.axhline(y, color="#CC3311", ls="--", lw=0.8)
    ax.set_xlabel("Module Membership (kME)")
    ax.set_ylabel(f"Gene Significance ({trait.split(' (')[0]})")
    ax.set_title(f"{tag}\nMM-GS r={mmgs_r:+.2f} | module r={r:+.2f} (FDR={fdr:.3f})",
                 fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    return dict(module=module, power=power, r2=r2, mt_r=float(r), mt_p=float(p),
                mt_fdr=float(fdr), mmgs_r=mmgs_r, mmgs_p=mmgs_p,
                hubs=list(hubs.index))
